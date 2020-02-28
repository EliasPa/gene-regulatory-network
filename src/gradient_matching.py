import numpy as np
from matplotlib import pyplot as plt


def return_design_matrix(X):
    return np.concatenate((np.ones(shape=(1, 19)), X))


# Copied from the exercise 5


def gradient_matching(T, X):
    names = ["SWI5", "CBF1", "GAL4", "GAL80", "ASH1"]

    s = "%9s   %11s   %11s   %15s   %18s   %18s   %18s   %18s" % (
        "Component", "Basal level", "Degradation", "Regulation by SWI5", "Regulation by CBF1", "Regulation by GAL4", "Regulation by GAl80", "Regulation by ASH1")
    print(s)
    for n in range(5):
        dxdt = np.diff(X[n, :]) / np.diff(T)
        H = return_design_matrix(X[:, 0:-1])
        beta = np.linalg.solve(H @ H.T, H @ dxdt)
        regs = np.copy(beta[1:6])
        print(names[n], beta)
        regs[n] = 0
        s = "{:^9s}   {:10.2f}     {:10.2f}    {:10.2f}            {:^10.2f}         {:^10.2f}           {:^10.2f}       {:^10.2f}".format(
            names[n], beta[0], -beta[n+1], regs[0], regs[1], regs[2], regs[3], regs[4])
        print(s)
    print("––––––––––––––––––––––––––––––––––––")


def gradient_matching_model_2(T, X):
    names = ["SWI5", "CBF1", "GAL4", "GAL80", "ASH1"]
    # Interacting genes
    # SWI5: CBF1, GAL4, GAL80, ASH1
    indices_SWI5 = [1, 2, 3, 4]
    # SBF1: SWI5, GAL4, GAL80, ASH1
    indices_CBF1 = [0, 2, 3, 4]
    # GAL4: GAL80
    indices_GAL4 = [3]
    # GAL80: GAL4, ASH1
    indices_GAL80 = [2, 4]
    # ASH1: SWI5, CBF1, GAL4, GAL80
    indices_ASH1 = [0, 1, 2, 3]

    interaction_indices = [indices_SWI5, indices_CBF1,
                           indices_GAL4, indices_GAL80, indices_ASH1]
    for n in range(5):
        dxdt = np.diff(X[n, :]) / np.diff(T)
        H = return_design_matrix(np.take(
            X[:, 0:-1], interaction_indices[n], axis=0))
        beta = np.linalg.solve(H @ H.T, H @ dxdt)
        # regs = np.copy(beta[1:6])
        print("Gene: ", names[n])
        print("––––––––––––––––––––––––––––––––––––")
        print("Bias", beta[0])
        for i, b in enumerate(beta[1:]):
            print("Effect of {}: {}".format(
                names[interaction_indices[n][i]], b))
        print("––––––––––––––––––––––––––––––––––––")


def gradient_matching_model_3(T, X):
    names = ["SWI5", "CBF1", "GAL4", "GAL80", "ASH1"]
    # Interacting genes
    # SWI5: GAL4, GAL80
    indices_SWI5 = [2, 3]
    # SBF1: SWI5, GAL4, GAL80, ASH1
    indices_CBF1 = [0, 3]
    # GAL4: GAL80
    indices_GAL4 = [3]
    # GAL80: GAL4, ASH1
    indices_GAL80 = [2, 4]
    # ASH1: SWI5, CBF1, GAL4, GAL80
    indices_ASH1 = [2, 3]

    interaction_indices = [indices_SWI5, indices_CBF1,
                           indices_GAL4, indices_GAL80, indices_ASH1]
    for n in range(5):
        dxdt = np.diff(X[n, :]) / np.diff(T)
        H = return_design_matrix(np.take(
            X[:, 0:-1], interaction_indices[n], axis=0))
        beta = np.linalg.solve(H @ H.T, H @ dxdt)
        # regs = np.copy(beta[1:6])
        print("Gene: ", names[n])
        print("––––––––––––––––––––––––––––––––––––")
        print("Bias", beta[0])
        for i, b in enumerate(beta[1:]):
            print("Effect of {}: {}".format(
                names[interaction_indices[n][i]], b))
        print("––––––––––––––––––––––––––––––––––––")
