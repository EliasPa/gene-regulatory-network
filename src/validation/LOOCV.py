import numpy as np
np.random.seed(123)
    
def loss(y_test, x_test, beta):
    """
    Computes the squared loss of y_test fitted with x_test * beta
    """
    n = beta.size - 1
    prediction =  np.append(1, x_test) @ beta
    return 0.5 * np.sum((y_test - prediction)**2)

def poly_loocv(T, X):
    """
    Performs leave-one-out crossvalidation.
    Loops over the elements of (y, x) and regresses
    that element on the other elements.

    Implementation base from exercise round 5 solutions.
    """
    gene_losses = []
    for n, gene in enumerate(X):
        loss_sum = 0
        for idx in range(gene.size-1):
            dxdt_test = (np.diff(gene) / np.diff(T))[idx]

            x_test = X[:, idx]

            t_train = np.delete(T, idx)
            x_train = np.delete(X, idx, axis=1)

            beta = polyfit(t_train, x_train, n)
            loss_sum += loss(dxdt_test, x_test, beta)
        
        gene_losses.append(loss_sum / gene.size)
    return (gene_losses, sum(gene_losses))

def create_design_full_model(x):
    """
    Creates a design matrix for the full model
    """
    ones = np.ones(shape=(1, x.shape[1] - 1))
    H = np.concatenate((ones, x[:, 0:-1]))

    return H

def polyfit(T, X, n):
    """
    Solves the polynomial regression
     y = beta_0 + x * beta_1 + ... + x^n beta_n
    Returns a vector [beta_0, ..., beta_n]
    """
    #print("makeV called from polyfit")
    H = create_design_full_model(X)
    #print("T", T.shape, "X",x.shape)
    # print(n)
    dxdt = np.diff(X[n]) / np.diff(T)
    #print("dX/dt:",dxdt.shape)
    #print("H, design:", H)
    beta = np.linalg.solve(H @ H.T, H @ dxdt)
    return beta
