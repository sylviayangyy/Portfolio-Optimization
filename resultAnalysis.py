import matplotlib.pyplot as plt
import numpy as np

def SharpeRatio(ret_value, risk_value, time_period=365, annual_risk_free_rate=0.032):
    risk_free = annual_risk_free_rate / 365 * time_period
    risk_free = 1.6
    # risk-free rate represents the interest an investor would expect from an absolutely risk-free investment over a specified period of time
    sharpe = (ret_value - risk_free) / risk_value
    return sharpe

def visualize(ret_value, risk_value, gamma, save2png=True, filename="figures/ret_risk.png"):
    # ret_value and risk_value are all vectors, each element corresponds to different gamma
    sharpe = SharpeRatio(ret_value, risk_value)
    # print(sharpe)
    
    plt.scatter(risk_value, ret_value, c=sharpe, cmap="OrRd")
    cbar = plt.colorbar()
    cbar.set_label('Sharpe Ratio', rotation=270, labelpad=+15)
    plt.xlabel('Risk')
    plt.ylabel('Return')
    max_sharpe_ret = ret_value[np.argmax(sharpe)]
    max_sharpe_risk = risk_value[np.argmax(sharpe)]
    max_sharpe_gamma = gamma[np.argmax(sharpe)]
    min_risk_ret = ret_value[np.argmin(risk_value)]
    min_risk_risk = risk_value[np.argmin(risk_value)]
    min_risk_gamma = gamma[np.argmin(risk_value)]
    plt.scatter(max_sharpe_risk, max_sharpe_ret, color='b', s=120)
    annotation = "       Maximum Sharpe ratio = " + "{0:.2f}".format(np.max(sharpe)) + "\n       gamma = " + "{0:.2f}".format(max_sharpe_gamma)
    plt.annotate(annotation, (max_sharpe_risk, max_sharpe_ret))
    plt.scatter(min_risk_risk, min_risk_ret, color='g', s=120)
    annotation = "     Minimum Risk = " + "{0:.2f}".format(np.min(risk_value)) + "\n     gamma = " + "{0:.2f}".format(min_risk_gamma)
    plt.annotate(annotation, (min_risk_risk, min_risk_ret))

    plt.savefig(filename)

if __name__ == "__main__":
    np.random.seed(2)
    n = 100
    ret_value = np.abs(np.random.randn(n))
    risk_value = np.abs(np.random.randn(n))+0.5
    gamma = np.abs(np.random.randn(n))
    visualize(ret_value, risk_value, gamma)