#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import joblib
import contextlib
from tqdm import tqdm
from joblib import Parallel, delayed

import numpy as np
import itertools
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.integrate import quad
from scipy.special import gamma
import math


# In[ ]:


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback

    try:
        yield tqdm_object

    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


# In[ ]:


TARG_LST = [
    'double_well',
    'gaussian',
    'gaussian_mixture'
]

ALGO_LST = [
    'HOLA',
    'mHOLA'
]

class LangevinSampler:
    def __init__(self, targ, algo, step=0.001, beta=1, a=None, d=None):
        assert targ in TARG_LST
        assert algo in ALGO_LST
        
        self.targ = targ
        self.algo = algo
        self.step = step
        self.beta = beta
        self.a = a  # for mixed gaussian
        self.d = d  # dimension of the parameter
            
    
    def _gradient(self, theta):
        if self.targ == 'double_well':
            return (np.dot(theta, theta) - 1) * theta
        elif self.targ == 'gaussian':
            return theta
        elif self.targ == 'gaussian_mixture':
            return theta - self.a + 2 * self.a / (1 + np.exp(2 * np.dot(theta, self.a)))
                
    def _hessian(self,theta):
        if self.targ == 'double_well':
            return (np.dot(theta, theta) - 1) * np.eye(self.d) + 2 * np.outer(theta, theta)
        elif self.targ == 'gaussian':
            return np.eye(self.d)
        elif self.targ == 'gaussian_mixture':
            return np.eye(self.d) - 4 * np.outer(self.a, self.a) * np.exp(2 * np.dot(theta, self.a)) / (1 + np.exp(2 * np.dot(theta, self.a)))**2
    
    def _vectorlaplacian(self, theta):
        if self.targ == 'double_well':
            return 2 * (self.d + 2) * theta
        elif self.targ == 'gaussian':
            return np.zeros(self.d)
        elif self.targ == 'gaussian_mixture':
            return 8 * self.a * np.dot(self.a, self.a) * (np.exp(
                4 * np.dot(theta, self.a)) - np.exp(2 * np.dot(theta, self.a))) / (1 + np.exp(2 * np.dot(theta, self.a)))**3
    
    def _gradient_tamed(self, theta):
        if self.algo == 'HOLA':
            return self._gradient(theta)
        elif self.algo == 'mHOLA':
            return self._gradient(theta) / ((1 + (self.step**(3/2))*(np.dot(theta, theta)**3))**(1/3))

    def _hessian_tamed(self,theta):
        if self.algo == 'HOLA':
            return self._hessian(theta)
        elif self.algo == 'mHOLA':
            return self._hessian(theta) / ((1 + (self.step**(3/2))*(np.dot(theta, theta)**3))**(1/3))

    def _vectorlaplacian_tamed(self,theta):
        if self.algo == 'HOLA':
            return self._vectorlaplacian(theta)
        elif self.algo == 'mHOLA':
            return self._vectorlaplacian(theta) / ((1 + (self.step**(3/2))*(np.dot(theta, theta)**3))**(1/3))

    
    def sample(self, theta0, n_iter=10**5, n_burnin=10**4, return_arr=False, runtime=200):
        if runtime is not None:
            n_iter = int(runtime/self.step)
            n_burnin = n_iter
            
        theta = np.ravel(np.array(theta0).reshape(-1))
        
        if return_arr:
            theta_arr = np.zeros((n_iter, self.d))

        for n in np.arange(n_iter + n_burnin):
            theta = theta  + self.step * (- self._gradient_tamed(theta) \
                                          + (self.step / 2) * (np.dot(self._hessian_tamed(theta), self._gradient_tamed(theta)) \
                                                               - self._vectorlaplacian_tamed(theta) / self.beta)) + np.sqrt(
                2 * self.step / self.beta) * (np.dot((np.eye(self.d) - self.step * self._hessian_tamed(theta) / 2), np.random.standard_normal(self.d)) \
                                             + (np.sqrt(3) / 6) * self.step * np.dot(self._hessian_tamed(theta), np.random.standard_normal(self.d)))
            if (n >= n_burnin) and return_arr:
                theta_arr[n - n_burnin] = theta

        return theta if (not return_arr) else theta_arr


# In[ ]:


def draw_samples_parallel(sampler, theta0, runtime=200, n_chains=100, n_jobs=-1):
    d = len(np.ravel(np.array(theta0).reshape(-1)))
    sampler.d = d
    def _run_single_markov_chain():
        return pd.DataFrame(
            [sampler.sample(theta0, runtime=runtime)],
            columns=[f'component_{i + 1}' for i in range(d)]
        )
        
    samples_df_lst = Parallel(n_jobs=n_jobs)(
        delayed(_run_single_markov_chain)() for i in tqdm(range(n_chains), desc='Markov Chains')
    )

    return pd.concat(samples_df_lst, ignore_index=True)


# In[ ]:


step_sizes = [0.00001, 0.005, 0.01, 0.025, 0.05, 0.1]
results_dict = {}
d = 100

# Loop through each step size and run the Langevin sampler
for step in step_sizes:
    print(f"Running Langevin sampler for step size: {step}")
    
    # Create a LangevinSampler instance with the specified step size
    sampler = LangevinSampler(targ='double_well', algo='mHOLA', step=step)
    
    # Initial theta values
    theta0 = np.zeros(d)
    
    # Draw samples in parallel
    results_df = draw_samples_parallel(sampler, theta0, n_chains=250)
    
    # Store the results DataFrame in the dictionary with the step size as the key
    results_dict[step] = results_df


# In[ ]:


def calculate_y(x):
    # Define the integrands for the numerator and denominator
    def integrand_denominator(r, d):
        return r ** (d / 2 - 1) * np.exp(-(1/4) * r**2 + (r)/2)

    # Calculate the integrals
    denominator = quad(integrand_denominator, 0, np.inf, args=(d,))[0]
    
    def integrand_numerator(r):
        return (r ** ((d - 3) / 2) * np.exp(-(1/4) * (r+x**2)**2 + (r+x**2)/2))
    
    numerator = quad(integrand_numerator, 0, np.inf)[0]

    y = (gamma(d / 2) / (np.sqrt(np.pi) * gamma((d - 1) / 2))) * numerator / denominator
    return y

component_1_data = results_dict[0.01]['component_1']

# Plot a histogram for the first component (component_1)
plt.hist(component_1_data, bins=10, density=True, color='skyblue', edgecolor='black')

# Superimpose the function calculate_y(x) on the histogram plot
x = np.linspace(-5, 5, 1000)  # Define the x-values for the function plot
y = [calculate_y(xi) for xi in x]  # Calculate the y-values
plt.plot(x, y, color='red', label='y(x)')

# Add title, labels, and legend
plt.title('Histogram of Component 1 with y(x) Superimposed')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()

# Show the plot
plt.show()

