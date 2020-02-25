import numpy as np

np.random.seed(123)
def loss_function(yi,x_pred):
    return (0.5)*(yi - x_pred)**2

def predict(x, betas, k):
    b0 = betas[0]
    S = np.zeros(x.shape[0])
    for i in range(1, k):
        S += betas[i]*(x**i)

    return np.array(b0 + S + np.random.random())

def LOOCV(x, T, k_max):
    losses = []
    for k in range(1, k_max + 1):
        model_losses = []
        for i, xi in enumerate(x):
            x_train = np.delete(x, i)
            y_train = np.delete(T, i)
            yi = T[i]

            # design matrix and beta calculation copied
            # from exercise 5 solutions:
            dxdt = np.diff(x_train[k, :]) / np.diff(y_train)
            H = np.concatenate((np.ones(shape=(1, 19)), x_train[:, 0:-1]))
            beta = np.linalg.solve(H @ H.T, H @ dxdt)

            x_pred = predict(x, beta, k)[i]
            model_losses.append(loss_function(yi, x_pred))
        losses.append(np.mean(model_losses))
    return losses
    
def loss(y_test, x_test, beta):
    """
    Computes the squared loss of y_test fitted with x_test * beta
    """
    n = beta.size - 1

    print("beta", beta)
    prediction =  np.append(1, x_test) @ beta
    print("prediction vs expected", prediction, y_test)
    return 0.5 * np.sum((y_test - prediction)**2)

def poly_loocv(T, X):
    """
    Performs leave-one-out crossvalidation.
    Loops over the elements of (y, x) and regresses
    that element on the other elements.
    """
    total_loss = 0
    for n, gene in enumerate(X):
        loss_sum = 0
        #print("gene", gene.shape)
        for idx in range(gene.size-1):
            dxdt_test = (np.diff(gene) / np.diff(T))[idx]

            x_test = X[:, idx]

            t_train = np.delete(T, idx)
            x_train = np.delete(X, idx, axis=1)

            beta = polyfit(t_train, x_train, n)
            loss_sum += loss(dxdt_test, x_test, beta)
        total_loss += loss_sum / gene.size
    return total_loss

def create_design(x):
    """
    Creates a design matrix of order n
    """
    print(x.shape)
    ones = np.ones(shape=(1, x.shape[1] - 1))
    H = np.concatenate((ones, x[:, 0:-1]))
    print("H in design:", H.shape)
    return H

def polyfit(T, X, n):
    """
    Solves the polynomial regression
     y = beta_0 + x * beta_1 + ... + x^n beta_n
    Returns a vector [beta_0, ..., beta_n]
    """
    #print("makeV called from polyfit")
    H = create_design(X)
    #print("T", T.shape, "X",x.shape)
    print(n)
    dxdt = np.diff(X[n]) / np.diff(T)
    print("dX/dt:",dxdt.shape)
    print("H, design:", H)
    beta = np.linalg.solve(H @ H.T, H @ dxdt)
    return beta
