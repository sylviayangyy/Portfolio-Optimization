from dataProcessing import *
import pandas as pd

from resultAnalysis import visualize
from solver import *

if __name__ == "__main__":
    period = 10

    # select stocks using 2015 data
    prices, _, _, stock = readCSVs(period=period)
    sectorList = ['Consumer Discretionary', 'Consumer Staples', 'Energy', 'Financials', 'Health Care', 'Industrials',
                  'Information Technology', 'Materials', 'Real Estate', 'Telecommunications Services', 'Utilities']
    stockName = []
    for i in sectorList:
        stockName += selectStocksFromSector(stock, i, n=2)
    # print(stockName)
    # calculate mu and sigma using 2015 data
    mu, Sigma = statistics(prices, stockName)

    # calculate return mean and std using 2016 data
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 1000)
    _, _, _, stock_2016 = readCSVs(period=period, start_date='2016-01-01', end_date='')
    # print(stock.head)
    # stock[return] is return mean, stock[risk] is return std
    # print(stock[stock['symbol']=='A']['return'].values[0])
    # print(stock[stock['symbol']=='A']['risk'].values[0])

    SAMPLES = 50
    ret_data = np.zeros(SAMPLES)
    risk_data = np.zeros(SAMPLES)
    # return mean
    # returns = np.zeros(SAMPLES)
    returns = []
    #  return std
    # risks = np.zeros(SAMPLES)
    risks = []
    w_data = []
    # gamma_data = np.zeros(SAMPLES)
    gamma_data = []
    gamma_vals = np.logspace(-2, 2, num=SAMPLES)
    for i in range(SAMPLES):
        ret_data[i], risk_data[i], w = solve(mu, Sigma, gamma=gamma_vals[i])
        # print(w)
        # print(w[0])
        w_data.append(w)

        assert len(stockName) == len(w)
        return_sum = 0
        risk_sum = 0
        for j in range(len(w)):
            return_sum += w[j] * stock_2016[stock_2016['symbol']==stockName[j]]['return'].values[0]
            risk_sum += w[j] * stock_2016[stock_2016['symbol']==stockName[j]]['risk'].values[0]
        if gamma_vals[i] < 20:
            # gamma_data[i] = gamma_vals[i]
            gamma_data.append(gamma_vals[i])
            # returns[i] = return_sum
            returns.append(return_sum)
            # risks[i] = risk_sum
            risks.append(risk_sum)
            if(return_sum==risk_sum):
                print("same")

    # for i in range(len(returns)):
    #     print(returns[i])
    #     print(risks[i])
    #     print(gamma_vals[i])
    #     print("\n")

    # return_data = np.zeros(len(returns))
    # risk_data = np.zeros(len(risks))
    # gamma_vals = np.zeros(len(gamma_data))
    # for i in range(len(returns)):
    #     return_data[i] = returns[i]
    #     risk_data[i] = risks[i]
    #     gamma_vals[i] = gamma_data[i]
    # visualize(stockName, return_data, risk_data, gamma_vals, w_data, period, filename="figures/test_result.png")
    plt.figure(figsize=(15, 10))
    plt.title('Final return vs Gamma')
    plt.xlabel('gamma')
    plt.ylabel('return')
    plt.plot(gamma_data, returns)
    plt.savefig("figures/Final_return_and_gamma.png")



