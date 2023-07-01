import numpy as np
import cvxpy as cp
import pandas as pd
import matplotlib.pyplot as plt

load_data = pd.read_csv('data/loads.csv', parse_dates=[0], index_col=0)["Load (kW)"]

# Split data into train set and test set
train = load_data[load_data.index.year <= 2021]
test = load_data[load_data.index.year > 2021]

def fit_baseline(x, lambd=50, tau=0.5):
    # Define constants and parameters
    periods = [period / harmonic for period in [24, 168, 8760] for harmonic in range(1, 5)]
    mu = np.array([1, 4, 9, 16]*3)
    K, T = len(periods), len(x)

    # Define variables
    alphas = cp.Variable(K)
    betas = cp.Variable(K)
    beta_0 = cp.Variable()

    # Define multi-sine seasonal baseline forecast
    t = np.arange(T)
    sines = np.array([np.sin(2 * np.pi * t / period) for period in periods]).T
    cosines = np.array([np.cos(2 * np.pi * t / period) for period in periods]).T
    
    b = beta_0 + sines @ alphas + cosines @ betas

    # Define loss function
    r = b - x
    loss = cp.sum(0.5 * cp.abs(r) + (tau - 0.5) * r) + lambd * cp.sum(cp.multiply(mu, alphas**2 + betas**2))

    # Specify and solve problem
    problem = cp.Problem(cp.Minimize(loss))
    problem.solve(solver=cp.MOSEK)
    
    return beta_0.value, alphas.value, betas.value

def make_baseline(T, beta_0, alphas, betas):
    # Define constants and parameters
    periods = [period / harmonic for period in [24, 168, 8760] for harmonic in range(1, 5)]
    mu = np.array([1, 4, 9, 16]*3)
    K = len(periods)
    
    # Define multi-sine seasonal baseline forecast
    t = np.arange(T)
    sines = np.array([np.sin(2 * np.pi * t / period) for period in periods]).T
    cosines = np.array([np.cos(2 * np.pi * t / period) for period in periods]).T

    
    b = beta_0 + sines @ alphas + cosines @ betas
    return b

def make_augmented_baseline(tau):
    # Fit baseline to training data
    beta_0, alphas, betas = fit_baseline(x=train.values, tau=tau)
    train_baseline = make_baseline(T=len(train.values), beta_0=beta_0, alphas=alphas, betas=betas)

    # Fit baseline to test data
    test_baseline = make_baseline(T=len(test.values), beta_0=beta_0, alphas=alphas, betas=betas)

    # Extend test baseline one year and save
    augmented_baseline = make_baseline(len(test.values)*2, beta_0, alphas, betas)
    date_range = pd.date_range(start='2022-01-01 00:00:00', end='2023-12-31 23:00:00', freq='H', tz=None)
    augmented_baseline_series = pd.Series(data=augmented_baseline, index=date_range)
    
    # Create pandas Series
    train_baseline_series = pd.Series(index=train.index, data=train_baseline)
    
    return pd.concat([train_baseline_series, augmented_baseline_series])

def fit_AR(tau, M, L, lambd=0.1):
    beta_0, alphas, betas = fit_baseline(train.values)
    train_baseline = make_baseline(len(train.values), beta_0, alphas, betas)
    train_baseline_series = pd.Series(index=train.index, data=train_baseline)

    train_residual = train - train_baseline

    train_hours = len(train_residual)

    past = np.zeros((M, train_hours - M + 1 - L))
    residuals = np.zeros((L, train_hours - M + 1 - L))

    for t in range(M-1, train_hours - L):
        i = t - M + 1
        residuals[:,i] = train_residual[t+1:t+1+L]
        past[:,i] = train_residual[t-M+1:t+1]

    AR_params = cp.Variable((L,M))
    error = AR_params @ past - residuals
    loss = cp.sum(0.5 * cp.abs(error) + (tau - 0.5) * error) + lambd * cp.sum_squares(AR_params)

    problem = cp.Problem(cp.Minimize(loss))
    problem.solve(solver=cp.MOSEK)
    
    return AR_params.value