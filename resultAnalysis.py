import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def CovHeatmap(Sigma, stockName):
    plt.figure(figsize = (25,20))
    sns.heatmap(Sigma, annot=True, cmap="Blues", xticklabels=stockName, yticklabels=stockName)
    plt.savefig('figures/SelectedStockCovariance.png')

def plotMu(mu, stockName):
    plt.figure(figsize = (24,8))
    ax = sns.barplot(x=mu.flatten(), y=stockName)
    plt.savefig('figures/stockExpectedReturn.png')

def SharpeRatio(ret_value, risk_value, time_period=365, annual_risk_free_rate=0.05):
    risk_free = annual_risk_free_rate / 365 * time_period
    # risk-free rate represents the interest an investor would expect from an absolutely risk-free investment over a specified period of time
    sharpe = (ret_value - risk_free) / risk_value
    return sharpe

def printStockInfo(stockName, weight):
    print("Stock Name | Proportion")
    for i in range(len(stockName)):
        print(stockName[i], "{}%".format(abs(weight[i]*100).round(3)))

def visualize(stockName, ret_value, risk_value, gamma, weight, time_period=365, save2png=True, filename="figures/ret_risk.png"):
    # ret_value and risk_value are all vectors, each element corresponds to different gamma
    sharpe = SharpeRatio(ret_value, risk_value, time_period)
    # print(sharpe)
    
    plt.figure(figsize=(15,10))
    plt.scatter(risk_value, ret_value, c=sharpe, cmap="OrRd")
    cbar = plt.colorbar()
    cbar.set_label('Sharpe Ratio', rotation=270, labelpad=+15)
    plt.xlabel('Risk')
    plt.ylabel('Return')

    # Max Sharpe Ratio
    index = np.argmax(sharpe)
    max_sharpe_ret = ret_value[index]
    max_sharpe_risk = risk_value[index]
    max_sharpe_gamma = gamma[index]
    plt.scatter(max_sharpe_risk, max_sharpe_ret, color='C2', s=120)
    annotation = "Maximum Sharpe ratio = " + "{0:.2f}".format(np.max(sharpe)) + "\ngamma = " + "{0:.2f}".format(max_sharpe_gamma)
    plt.annotate(annotation, (max_sharpe_risk, max_sharpe_ret))#, (max_sharpe_risk+0.2, max_sharpe_ret-0.05))

    print("To Achieve Maximum Sharpe Ratio: gamma = " + "{0:.2f}".format(max_sharpe_gamma))
    print("-------------------------------")
    printStockInfo(stockName, weight[index])

    # Min Risk
    index = np.argmin(risk_value)
    min_risk_ret = ret_value[index]
    min_risk_risk = risk_value[index]
    min_risk_gamma = gamma[index]
    plt.scatter(min_risk_risk, min_risk_ret, color='C0', s=120)
    annotation = "Minimum Risk = " + "{0:.2f}".format(np.min(risk_value)) + "\ngamma = " + "{0:.2f}".format(min_risk_gamma)
    plt.annotate(annotation, (min_risk_risk, min_risk_ret))#, (min_risk_risk+0.2, min_risk_ret))

    print("\n\nTo Achieve Minimum Risk: gamma = " + "{0:.2f}".format(min_risk_gamma))
    print("-------------------------------")
    printStockInfo(stockName, weight[index])

    plt.savefig(filename)

    

if __name__ == "__main__":
    np.random.seed(2)
    n = 100
    nstock = 10
    stockName = ["stock"+str(i) for i in range(nstock)]
    ret_value = np.abs(np.random.randn(n))
    risk_value = np.abs(np.random.randn(n))+0.5
    gamma = np.abs(np.random.randn(n))
    weight = np.abs(np.random.randn(n, nstock))
    visualize(stockName, ret_value, risk_value, gamma, weight)