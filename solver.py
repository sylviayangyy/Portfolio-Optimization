import numpy as np
import cvxpy as cp

def solve(mu, Sigma, gamma):
    # mu: nx1
    # Sigma: nxn
    # gamma>0: risk aversion parameter
    n = np.shape(mu)[0]
    assert n == np.shape(Sigma)[0]
    assert n == np.shape(Sigma)[1]
    assert gamma > 0

    w = cp.Variable(n)
    ret = mu.T * w
    risk = cp.quad_form(w, Sigma)
    prob = cp.Problem(cp.Maximize(ret - gamma * risk),
                      [cp.sum(w) == 1, w >= 0])
    prob.solve()
    ret_data = ret.value
    risk_data = cp.sqrt(risk).value
    w_data = w.value
    return ret_data, risk_data, w_data


