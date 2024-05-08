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
from scipy.stats import wasserstein_distance
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
    'ULA',
    'MALA',
    'TULA',
    'mTULA',
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

    
    def sample(self, theta0, n_iter=10**5, n_burnin=10**4, return_arr=True, runtime=200):
        if runtime is not None:
            n_iter = int(runtime/self.step)
            n_burnin = n_iter
            
        theta = np.ravel(np.array(theta0).reshape(-1))
        
        if return_arr:
            #theta_arr = np.zeros(n_iter)
            theta_arr = np.zeros(n_iter + n_burnin)

        for n in np.arange(n_iter + n_burnin):
            theta = theta  + self.step * (- self._gradient_tamed(theta) \
                                          + (self.step / 2) * (np.dot(self._hessian_tamed(theta), self._gradient_tamed(theta)) \
                                                               - self._vectorlaplacian_tamed(theta) / self.beta)) + np.sqrt(
                2 * self.step / self.beta) * (np.dot((np.eye(self.d) - self.step * self._hessian_tamed(theta) / 2), np.random.standard_normal(self.d)) \
                                             + (np.sqrt(3) / 6) * self.step * np.dot(self._hessian_tamed(theta), np.random.standard_normal(self.d)))
            #if (n >= n_burnin) and return_arr:
            if return_arr:
                theta_arr[n] = theta[0]

        return theta if (not return_arr) else theta_arr


# In[ ]:


def draw_samples_parallel(sampler, theta0, runtime=200, n_chains=100, n_jobs=-1):
    d = len(np.ravel(np.array(theta0).reshape(-1)))
    sampler.d = d
    def _run_single_markov_chain():
        theta_first_dimension_samples = sampler.sample(theta0, return_arr=True, runtime=runtime)
        return pd.DataFrame(
            theta_first_dimension_samples.reshape(1, -1),
            columns=[f'iteration_{i + 1}' for i in range(len(theta_first_dimension_samples))]
        )
        
    samples_df_lst = Parallel(n_jobs=n_jobs)(
        delayed(_run_single_markov_chain)() for i in tqdm(range(n_chains), desc='Markov Chains')
    )

    return pd.concat(samples_df_lst, ignore_index=True)


# In[ ]:


step_sizes = [0.01]
d = 100
theta0 = 2*np.ones(d)
no_chains = 500
dw_results_dict = {}
for step in step_sizes:
    print(f"Running Langevin sampl er for step size: {step}")
    
    # Create a LangevinSampler instance with the specified step size
    dw_sampler = LangevinSampler(targ='double_well', algo='mHOLA', step=step)
  
    # Draw samples in parallel
    dw_results_df = draw_samples_parallel(dw_sampler, theta0, n_chains=no_chains)
    
    # Store the results DataFrame in the dictionary with the step size as the key
    dw_results_dict[step] = dw_results_df


# In[ ]:


def calculate_y(x):
    # Define the integrands for the numerator and denominator
    def integrand_denominator(r):
        return r ** (d / 2 - 1) * np.exp(-(1/4) * r**2 + (r)/2)

    # Calculate the integrals
    denominator = quad(integrand_denominator, 0, np.inf)[0]

    def integrand_numerator(r):
        return (r ** ((d - 3) / 2) * np.exp(-(1/4) * (r+x**2)**2 + (r+x**2)/2))

    numerator = quad(integrand_numerator, 0, np.inf)[0]

    y = (gamma(d / 2) / (np.sqrt(np.pi) * gamma((d - 1) / 2))) * numerator / denominator
    return y

# Number of samples to generate
n_dw_samples = no_chains

# Generate samples using the inverse transform method
def generate_samples(n):
    samples = []
    for _ in range(n):
        accepted = False
        while not accepted:
            x = np.random.normal(0, 1)  # Generate a random value for x from a standard normal distribution
            u = np.random.uniform(0, 1)  # Generate a uniform random value for acceptance/rejection
            if u < calculate_y(x):
                samples.append(x)
                accepted = True
    return samples

# Generate the samples
dw_samples = generate_samples(n_dw_samples)


# In[ ]:


# Create a dictionary to store the computed distances
dw_distance_dict = {}

# Iterate over each column of the DataFrame stored in results_dict[step]
for column in dw_results_dict[step].columns:
    column_data = dw_results_dict[step][column].values  # Get the values of the column
    distance = wasserstein_distance(dw_samples, column_data)  # Compute the Wasserstein-1 distance
    dw_distance_dict[column] = distance  # Store the computed distance in the dictionary


# In[ ]:


class TULALangevinSampler:
    def __init__(self, targ, algo, step=0.001, beta=1, Sigma=None, a=None, alpha=None, lambd=None, tau=None):
        assert targ in TARG_LST
        assert algo in ALGO_LST

        self.targ = targ
        self.algo = algo
        self.beta = beta
        self.step = step
        self.adjust = (algo == 'MALA')

        self.Sigma = Sigma  # for gaussian target
        self.a = a  # for mixed gaussian target

        # ginzburg_landau parameters
        self.alpha = alpha
        self.lambd = lambd
        self.tau = tau


        if targ == 'double_well':
            self.r = 2

        elif targ == 'gaussian' or targ == 'gaussian_mixture':
            self.r = 0

        elif targ == 'ginzburg_landau':
            self.r = 2


    def _potential(self, theta):
        if self.targ == 'double_well':
            return (1 / 4) * np.dot(theta, theta)**2 - (1/2) * np.dot(theta, theta)

        elif self.targ == 'gaussian':
            return (1 / 2) * ((theta.dot(np.linalg.inv(self.Sigma))).dot(theta))

        elif self.targ == 'gaussian_mixture':
            return (1 / 2) * (np.linalg.norm(theta-self.a)**2) - np.log(1 + np.exp(-2*np.dot(theta, self.a)))

        elif self.targ == 'ginzburg_landau':
            theta_mat = theta.reshape([int(np.cbrt(len(theta)))]*3)

            return ((1-self.tau)/2) * ((theta_mat**2).sum()) + \
                   (self.tau * self.lambd/4) * ((theta_mat**4).sum()) + \
                   (self.tau * self.alpha/2) * sum([((np.roll(theta_mat, -1, axis) - theta_mat) ** 2).sum() for axis in range(3)])


    def _gradient(self, theta):
        if self.targ == 'double_well':
            return (np.dot(theta, theta) - 1) * theta

        elif self.targ == 'gaussian':
            return np.linalg.inv(self.Sigma).dot(theta)

        elif self.targ == 'gaussian_mixture':
            return theta - self.a + 2 * self.a / (1 + np.exp(2*np.dot(theta, self.a)))

        elif self.targ == 'ginzburg_landau':
            theta_mat = theta.reshape([int(np.cbrt(len(theta)))]*3)
            grad_mat = (1-self.tau)*theta_mat + \
                       (self.tau * self.lambd)*(theta_mat**3) + \
                       (self.tau * self.alpha) * (
                            6 * theta_mat - \
                            np.roll(theta_mat, -1, axis=0) - \
                            np.roll(theta_mat, -1, axis=1) -
                            np.roll(theta_mat, -1, axis=2) - \
                            np.roll(theta_mat, 1, axis=0) - \
                            np.roll(theta_mat, 1, axis=1) - \
                            np.roll(theta_mat, 1, axis=2)
                        )
            return grad_mat.reshape(-1)


    def _gradient_tamed(self, theta):
        if self.algo == 'ULA' or self.algo == 'MALA':
            return self._gradient(theta)

        elif self.algo == 'TULA':
            grad = self._gradient(theta)
            return grad / (1 + (self.step) * np.linalg.norm(grad))

        elif self.algo == 'mTULA':
            return self._gradient(theta) / ((1 + self.step*(np.dot(theta, theta)**self.r))**0.5)


    def sample(self, theta0, n_iter=10**5, n_burnin=10**4, return_arr=True, runtime=200):
        # if runtime is specified
        if runtime is not None:
            # replace the number of iterations and burn-in samples such that step*n_iter is constant
            n_iter = int(runtime/self.step)
            n_burnin = n_iter

        # flatten array to 1d
        theta = np.ravel(np.array(theta0).reshape(-1))

        # obtain dimension
        d = len(theta)

        # initialise array to store samples after burn-in period
        if return_arr:
            theta_arr = np.zeros(n_iter + n_burnin)

        # run algorithm
        for n in np.arange(n_iter + n_burnin):

            # proposal
            proposal = theta - self.step * self._gradient_tamed(theta) + np.sqrt(
                2 * self.step / self.beta) * np.random.normal(size=d)

            # if metropolis-hastings version is run
            if self.adjust:
                # potential at current iteration and proposal
                U_proposal = self._potential(proposal)
                U_theta = self._potential(theta)

                # (tamed) gradient at current iteration and proposal
                h_proposal = self._gradient_tamed(proposal)
                h_theta = self._gradient_tamed(theta)

                # logarithm of acceptance probability
                log_acceptance_prob = -self.beta * (U_proposal - U_theta) + \
                                      (1 / (4 * self.step)) * (np.linalg.norm(
                    proposal - theta + self.step * h_theta)**2 - np.linalg.norm(
                    theta - proposal + self.step * h_proposal)**2)

                # determine acceptance
                if np.log(np.random.uniform(size=1)) <= log_acceptance_prob:
                    theta = proposal

            # if not, then an unadjusted version is run
            else:
                theta = proposal

            # include samples after burn-in in final output
            if return_arr:
                theta_arr[n] = theta[0]

        return theta if (not return_arr) else theta_arr


# In[ ]:


dw_tula_results_dict = {}
for step in step_sizes:
    print(f"Running Langevin sampl er for step size: {step}")
    
    # Create a LangevinSampler instance with the specified step size
    dw_tula_sampler = TULALangevinSampler(targ='double_well', algo='mTULA', step=step)
    
    # Draw samples in parallel
    dw_tula_results_df = draw_samples_parallel(dw_tula_sampler, theta0, n_chains=no_chains)
    
    # Store the results DataFrame in the dictionary with the step size as the key
    dw_tula_results_dict[step] = dw_tula_results_df


# In[ ]:


# Create a dictionary to store the computed distances
dw_tula_distance_dict = {}

# Iterate over each column of the DataFrame stored in results_dict[step]
for column in dw_tula_results_dict[step].columns:
    column_data = dw_tula_results_dict[step][column].values  # Get the values of the column
    distance = wasserstein_distance(dw_samples, column_data)  # Compute the Wasserstein-1 distance
    dw_tula_distance_dict[column] = distance  # Store the computed distance in the dictionary


# In[ ]:


# Get the column names and distances from the distance_dict
distances = list(dw_distance_dict.values())
tula_distances = list(dw_tula_distance_dict.values())

# Create the first plot
fig, axs = plt.subplots(1, 2, figsize=(15, 6))  # Create a subplot with 1 row and 2 columns
axs[0].plot( distances, label='aHOLA')  # Plot aHOLA line
axs[0].plot(tula_distances, alpha=0.7,label='mTULA')  # Plot mTULA line
axs[0].set_ylabel('$W_1$ distance between $\pi_{\\beta}$ and the law of aHOLA and mTULA')
axs[0].set_xlabel('Number of Iterations')
#axs[0].set_title('Wasserstein-1 Distance for Each Iteration')
axs[0].legend()  # Show legend

# Get the last 100 values from dw_tula_distance_dict.values()
tula_distances_last_100 = list(dw_tula_distance_dict.values())[39900:]
ahola_distances_last_100 = list(dw_distance_dict.values())[39900:]

# Create the second plot
axs[1].plot(ahola_distances_last_100, linestyle='-', label='aHOLA')
axs[1].plot(tula_distances_last_100, linestyle='-', label='mTULA')
axs[1].set_xlabel('Number of Iterations')
#axs[1].set_title('Wasserstein-1 Distance for Last 100 Iterations')
axs[1].legend() 
axs[1].set_ylim(0, 0.15)
plt.tight_layout() 

plt.savefig('dw_W1_v2.png')
plt.show()


# In[ ]:


# Create a dictionary to store the computed distances
diff = {}

# Iterate over each column of the DataFrame stored in results_dict[step]
for column in dw_tula_results_dict[step].columns:
    diff_data = dw_distance_dict[column] - dw_tula_distance_dict[column]  # Get the values of the difference
    diff[column] = diff_data  # Store the computed distance in the dictionary


# In[ ]:


plt.plot(list(diff.values()))
plt.show()


# In[ ]:


list1 = list(diff.values())
pos_count, neg_count = 0, 0

for num in list1:
   if num >= 0:
      pos_count += 1
   else:
      neg_count += 1
print("Positive numbers in the list: ", pos_count)
print("Negative numbers in the list: ", neg_count)


# ### 
