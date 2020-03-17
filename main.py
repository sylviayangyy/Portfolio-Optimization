# case 1: we calculate the customer's personalized gamma value based on a survey
# case 2: varying gamma to find the optimal risk-return trade-of (sharpe ratio)

from solver import *
from resultAnalysis import *
import numpy as np

# TODO: select n stocks form the dataset
# TODO: generate mu and Sigma based on dataset
np.random.seed(2)
n = 10 # the number of stocks the customer wants to hold at the same time
mu = np.abs(np.random.randn(n, 1)) # expected return of each stock
Sigma = np.random.randn(n, n)
Sigma = Sigma.T.dot(Sigma) # Sigma is a postive semi-definite symmetric matrix

SAMPLES = 50
ret_data = np.zeros(SAMPLES)
risk_data = np.zeros(SAMPLES)
w_data = []

gamma_vals = np.logspace(-2, 0.5, num=SAMPLES)
for i in range(SAMPLES):
    ret_data[i], risk_data[i], w = solve(mu, Sigma, gamma=gamma_vals[i])
    print("Gamma = ", gamma_vals[i])
    print("w = ", w)
    print("return = ", ret_data[i], "\trisk = ", risk_data[i])
    w_data.append(w)

visualize(ret_data, risk_data, gamma_vals)