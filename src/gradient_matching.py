import numpy as np
from matplotlib import pyplot as plt


# Copied from the exercise 5
def gradient_matching(T, X):
    names = ["SWI5", "CBF1", "GAL4", "GAL80", "ASH1"]

    s = "%9s   %11s   %11s   %15s   %18s   %18s   %18s   %18s" % (
        "Component", "Basal level", "Degradation", "Regulation by SWI5", "Regulation by CBF1", "Regulation by GAL4", "Regulation by GAl80", "Regulation by ASH1")
    print(s)
    print(X, X.shape)
    constants = np.ones(shape=(5, 19))
    H = np.concatenate((np.ones(shape=(1, 19)), X[:, 0:-1]))
    for n in range(5):
        print("X and T: ", X[n,:].shape,T.shape)
        dxdt = np.diff(X[n, :]) / np.diff(T)
        H = np.concatenate((np.ones(shape=(1, 19)), X[:, 0:-1]))
        beta = np.linalg.solve(H @ H.T, H @ dxdt)
        regs = np.copy(beta[1:6])
        regs[n] = 0
        s = "{:^9s}   {:10.2f}     {:10.2f}    {:10.2f}            {:^10.2f}         {:^10.2f}           {:^10.2f}       {:^10.2f}".format(
            names[n], beta[0], -beta[n+1], regs[0], regs[1], regs[2], regs[3], regs[4])
        print(s)
