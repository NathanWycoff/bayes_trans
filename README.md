

import numpy as np
#exec(open("python/bayes_lib.py").read())
from bayes_trans import bayes_trans

# Synthetic data settings
sigma = 0.5
Ns = [5, 20, 20]
Ntest = 1000
P = 4
K = len(Ns)-1 # Number of source datasets, so one less than total number.

# Assume shared coef.
beta_targ = np.random.normal(size=P)
beta_other = np.random.normal(size=P)
betas = [beta_targ, beta_targ, beta_other]

# Generate data
Xs = [np.random.normal(size=[Ns[k],P]) for k in range(K)]
XX0 = np.random.normal(size=[Ns[0],P])
ys = [Xs[k] @ betas[k] + sigma*np.random.normal(size=Ns[k]) for k in range(K)]

# !!! Make sure your real X and y data have been centered and scaled ("z-scored")!!!

# Fit - takes about 2 minutes on our machine. Code is optimized for larger datasets.
beta0_trans, tracking, switch_rates = bayes_trans(Xs, ys)

# beta0_trans - Estimate of target data coefficients.
# tracking - MCMC trace for all variables.
# switch_rates - Parellel tempering exchange proposal acceptance rates.

beta0_naive = np.linalg.lstsq(Xs[0], ys[0])[0]

print(f"OLS Estimation Error: {np.sum(np.square(beta0_naive-betas[0]))}")
print(f"Transfer Learning Estimation Error: {np.sum(np.square(beta0_trans-betas[0]))}")


