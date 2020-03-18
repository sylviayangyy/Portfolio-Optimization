import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def readCSVs(period=1, annual_risk_free_rate=0.05):
    prices = pd.read_csv('nyse/prices-split-adjusted.csv')
    securities = pd.read_csv('nyse/securities.csv')
    fundamentals = pd.read_csv('nyse/fundamentals.csv')

    securities = securities.rename(columns = {'Ticker symbol' : 'symbol','GICS Sector' : 'sector'})
    prices = prices.merge(securities[['symbol', 'sector']], on='symbol')
    prices = prices[prices['date'] >= '2015-01-01'] 
    prices = prices[prices['date'] < '2016-01-01']
    # only consider data in 2015

    sector_pivot = pd.pivot_table(prices, values='close', index=['date'], columns=['sector']).reset_index()

    prices['return'] = prices.close / prices.close.shift(period) - 1 # Return on Investment: Net Profit / Cost of Investment
    prices = prices.drop(prices[prices['symbol'] != prices['symbol'].shift(period)].index)
    prices.dropna(inplace=True)

    risk_free = annual_risk_free_rate / 365 * period
    # print(risk_free)
    sector = pd.DataFrame({'return' : (prices.groupby('sector')['return'].mean()), 'risk' : prices.groupby('sector')['return'].std()})
    sector['sharpe'] = (sector['return'] - risk_free) / sector['risk']

    stock = pd.DataFrame({'return' : (prices.groupby('symbol')['return'].mean()), 'risk' : prices.groupby('symbol')['return'].std()})
    stock = stock.merge(securities[['symbol', 'sector']], on='symbol')
    stock['sharpe'] = (stock['return'] - risk_free) / stock['risk']
    
    return prices, sector_pivot, sector, stock

def selectStocksFromSector(stock, sector, n=2):
    stocksinSector = stock[stock['sector']==sector]
    sharpe = stocksinSector.sort_values(by=['sharpe'], ascending=False)
    # print(sharpe.head())

    sharpe = sharpe[:n]

    selectedStocks = sharpe['symbol'].values.tolist()

    return selectedStocks

def statistics(prices, stockName):
    priceList = []
    for stock in stockName:
        subset = prices[prices['symbol'] == stock]
        stockPrice = subset['return']
        priceList.append(stockPrice)
    priceList = np.array(priceList)
    # print(np.shape(priceList))

    mu = np.mean(priceList, axis=1)
    mu = np.reshape(mu,(mu.size, 1))

    Sigma = np.cov(priceList)

    return mu, Sigma

if __name__ == "__main__":
    prices, sector_pivot, sector, stock = readCSVs(period=10)
    print(prices.head())
    print(sector_pivot.head())
    print(sector.head())
    print(stock.head())
    plt.figure(figsize=(15, 6))
    ax = sns.countplot(y='sector', data=prices)
    plt.xticks(rotation=45)
    plt.savefig('figures/GICSsectorCount.png')

    plt.figure(figsize = (10,10))
    sns.heatmap(sector_pivot.corr(),annot=True, cmap="Blues")
    plt.savefig('figures/SectorCovariance.png')

    '''
    When building a diversified portfolio, investors seek negatively correlated stocks. Doing so reduces the risk of catastrophic losses in the portfolio and helps the investor sleep better at night. Assume the portfolio consists of two stocks and they are negatively correlated. This implies that when the price of one performs worse than usual, the other will likely do better than usual. However, risk takers would love to seek for positively correlated stocks for higher expected return, and of course, higher risk.
    '''

    plt.figure(figsize = (24,8))
    ax = sns.barplot(x=sector['sharpe'], y=sector.index)
    plt.savefig('figures/SectorSharpe.png')

    sectorList = ['Energy', 'Financials', 'Industrials', 'Information Technology','Materials', 'Utilities']
    selectedStocks = []
    for i in sectorList:
        selectedStocks += selectStocksFromSector(stock, i, n=3)
    print(selectedStocks)

    mu, Sigma = statistics(prices, selectedStocks)
    print(np.shape(mu))
    print(np.shape(Sigma))
