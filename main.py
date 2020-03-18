# case 1: we calculate the customer's personalized gamma value based on a survey
# case 2: varying gamma to find the optimal risk-return trade-of (sharpe ratio)

from solver import *
from resultAnalysis import *
from dataProcessing import *
import numpy as np

n = 22 # the number of stocks the customer wants to hold at the same time
period = 10

prices, _, _, stock = readCSVs(period=period)
sectorList = ['Consumer Discretionary', 'Consumer Staples', 'Energy', 'Financials', 'Health Care', 'Industrials', 'Information Technology','Materials', 'Real Estate', 'Telecommunications Services', 'Utilities']
stockName = []
for i in sectorList:
    stockName += selectStocksFromSector(stock, i, n=2)
print(stockName)

mu, Sigma = statistics(prices, stockName)
# mu = np.abs(np.random.randn(n, 1))
# Sigma = np.random.randn(n, n)
# Sigma = Sigma.T.dot(Sigma)
CovHeatmap(Sigma, stockName)
plotMu(mu, stockName)

SAMPLES = 50
ret_data = np.zeros(SAMPLES)
risk_data = np.zeros(SAMPLES)
w_data = []

gamma_vals = np.logspace(-2, 2, num=SAMPLES)
for i in range(SAMPLES):
    ret_data[i], risk_data[i], w = solve(mu, Sigma, gamma=gamma_vals[i])
    # print("Gamma = ", gamma_vals[i])
    # print("w = ", w)
    # print("return = ", ret_data[i], "\trisk = ", risk_data[i])
    w_data.append(w)

visualize(stockName, ret_data, risk_data, gamma_vals, w_data, period)

given_gamma = 5
_, _, ww = solve(mu, Sigma, gamma=given_gamma)

print("\n\nIf given gamma = " + "{0:.2f}".format(given_gamma))
print("-------------------------------")
printStockInfo(stockName, ww)