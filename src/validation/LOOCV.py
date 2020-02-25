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
    
def loss(t_test, x_test, beta):
    """
    Computes the squared loss of y_test fitted with x_test * beta
    """
    n = beta.size - 1
    #print("makeV called from loss", t_test, x_test)
    print("beta",beta)
    prediction = makeV(np.array([x_test]), n).T @ beta
    print("prediction", prediction)
    return 0.5 * np.sum((x_test - prediction)**2)

def poly_loocv(T, X, n):
    """
    Performs leave-one-out crossvalidation.
    Loops over the elements of (y, x) and regresses
    that element on the other elements.
    """
    total_loss = 0
    for gene in X:
        loss_sum = 0
        #print("gene", gene.shape)
        for idx in range(gene.size):
            t_test = gene[idx]
            x_test = gene[idx]

            t_train = np.delete(T, idx)
            x_train = np.delete(gene, idx)

            #print("X-train shape: ", x_train.shape)
            beta = polyfit(t_train, x_train, n)
            loss_sum += loss(t_test, x_test, beta)
            print(loss_sum)
        total_loss += loss_sum / gene.size
    return total_loss

def makeV(x, n):
    """
    Creates a design matrix of order n
    """
    #print((np.ones(shape=(1, x.shape[0])), x[0:-1]))

    # print("makeV called:", T.shape)
    x_2 = x
    
    #print("X in design:", x_2)
    #print("T in design:", T_2)
    #print(type(x_2))
    #if type(x_2) is not np.ndarray:
    #  print("x was not np.array")
    #  x_2 = np.array([x])
    #  T_2 = np.array([T])
    
    H = np.concatenate((np.ones(shape=(1, x_2.shape[0] - 1)), x_2[0:-1].reshape(1, x_2.shape[0]-1)))
    print("H in design:", H)
    return H

def polyfit(T, x, n):
    """
    Solves the polynomial regression
     y = beta_0 + x * beta_1 + ... + x^n beta_n
    Returns a vector [beta_0, ..., beta_n]
    """
    #print("makeV called from polyfit")
    H = makeV(x, n)
    #print("T", T.shape, "X",x.shape)
    dxdt = np.diff(x) / np.diff(T)
    #print("dX/dt:",dxdt.shape)
    #print("H, design:", H)
    beta = np.linalg.solve(H @ H.T, H @ dxdt)
    return beta
