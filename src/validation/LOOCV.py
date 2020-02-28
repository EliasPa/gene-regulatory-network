import numpy as np
from utils.utils import get_interaction_indices
np.random.seed(123)
    
def loss(y_test, x_test, beta):
    """
    Computes the squared loss of y_test fitted with x_test * beta
    """
    n = beta.size - 1
    prediction =  np.append(1, x_test) @ beta
    return 0.5 * np.sum((y_test - prediction)**2)

def poly_loocv(T, X, model):
    """
    Performs leave-one-out crossvalidation.
    Loops over the elements of all genes and regresses
    that element on the other elements.

    Implementation base from exercise round 5 solutions.
    """
    all_model_indices = get_interaction_indices(model)

    gene_losses = []
    for n, gene in enumerate(X):
        loss_sum = 0
        indices = all_model_indices[n]
        for idx in range(gene.size-1):
            dxdt_test = (np.diff(gene) / np.diff(T))[idx]

            x_test = np.take(X[:, idx], indices)

            t_train = np.delete(T, idx)
            x_train = np.delete(X, idx, axis=1)

            beta = polyfit(t_train, x_train, n, indices)
            loss_sum += loss(dxdt_test, x_test, beta)
        
        gene_losses.append(loss_sum / gene.size)
    return (sum(gene_losses), gene_losses)

def create_design(x, indices):
    """
    Creates a design matrix for the requested model
    (determined by indices)
    """
    data = np.take(x, indices, axis=0)
    ones = np.ones(shape=(1, data.shape[1] - 1))
    H = np.concatenate((ones, data[:, 0:-1]))

    return H

def polyfit(T, X, n, indices):
    """
    Solves the polynomial regression
     y = beta_0 + x * beta_1 + ... + x^n beta_n
    Returns a vector [beta_0, ..., beta_n]
    """
    H = create_design(X,indices)
    dxdt = np.diff(X[n]) / np.diff(T)
    beta = np.linalg.solve(H @ H.T, H @ dxdt)
    return beta
