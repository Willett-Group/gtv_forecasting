import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold


# NOTE: I eventually want to turn this into a class - sorry for all the separated functions for now!

"""
We define each piece of the GTV objective function we wish to minimize here. 
"""

def loss_fn(X, Y, beta, alpha=1):
    """
    Computes the loss function
    :param X: n x p matrix containing predictor values
    :param Y: n-dimensional vector of the response
    :param beta: p-dimensional vector of coefficients
    :param alpha: float between 0 and 1 defining the discount factor in the loss function
    :return: value of the loss function for given inputs
    """
    n = X.shape[0]
    if alpha<1:
        weights = np.array([alpha**(n-t) for t in np.arange(1, n+1)])
        X = X * np.sqrt(weights.reshape(-1,1))
        Y = Y * np.sqrt(weights)
    return 1/n*cp.norm2(cp.matmul(X, beta) - Y)**2

def regularizer(beta):
    """
    Lasso penalty term
    :param beta: p-dimensional vector of coefficients
    :return: \ell-1 norm of beta (sum of absolute value of elements)
    """
    return cp.norm1(beta)

def difference_pen(beta, edge_incidence):
    """
    GTV penalty term (see paper for more details)
    :param beta: p-dimensional vector of coefficients
    :param edge_incidence: |E| x p edge-incidence matrix representing the GTV term
    :return: GTV penalty value
    """
    return cp.norm1(edge_incidence @ beta)

def objective_fn(X, Y, beta, edge_incidence, lam1, lam2, alpha=1):
    """
    Full GTV objective function we want to minimize
    :param X: n x p matrix of predictors
    :param Y: n-dim response vector
    :param beta: p-dim coefficient vector
    :param edge_incidence: |E| x p edge-incidence matrix
    :param lam1: Lasso regularization parameter
    :param lam2: GTV regularization parameter
    :param alpha: discount penalty (we use 0.9 in the paper)
    :return: Value of GTV objective function for given inputs
    """
    return loss_fn(X, Y, beta, alpha) + lam1 * regularizer(beta) + lam2 * difference_pen(beta, edge_incidence)

def gtv_cvx(X, y, D, lambda_lasso, lambda_tv, alpha=1):
    """
    Function to actually estimate beta for specific values of regularization parameters
    :param X: n x p matrix of predictors
    :param y: n-dim response vector
    :param D: |E| x p edge-incdience matrix (same as edge_incidence in other functions)
    :param lambda_lasso: Lasso regularizatin parameter
    :param lambda_tv: GTV regularization parameter
    :param alpha: discount penalty between 0 and 1
    :return: estimated p-dim coefficient vector
    """
    # define the variables
    p = X.shape[1]
    beta = cp.Variable(p)
    lam1 = cp.Parameter(nonneg=True)
    lam2 = cp.Parameter(nonneg=True)
    # setup the problem
    problem = cp.Problem(cp.Minimize(objective_fn(X, y, beta, D, lam1, lam2, alpha)))
    # set regularization parameters
    lam1.value = lambda_lasso
    lam2.value = lambda_tv
    # solve the objective
    problem.solve()
    # return coefficients
    return beta.value

def gtv_cvx_path(X, y, D, lambda_lasso_path, lambda_tv_path,
                 return_df = True, region=None, alpha=1, t=50):
    """
    This function allows you to efficiently iterate over parameters and keep track of errors.
    Useful because if we are doing CV/many lambdas, shouldn't redefine the problem for each one
    because CVX defaults to doing a warm start
    :param X: n x p matrix of predictors
    :param y: n-dim response vector
    :param D: |E| x p edge-incdience matrix (same as edge_incidence in other functions)
    :param lambda_lasso_path: list of lasso regularization values to iterate over
    :param lambda_tv_path: list of GTV regularization values to iterate over
    :param return_df: If True this returns a dataframe, otherwise returns a list
    :param region: if you want the returned dataframe to keep track of the region you are working on, include here
    :param alpha: discount factor
    :param t: where you want the train/test split to take place (i.e. if your data is from 1940-2020, t=50
              means that you train on 1940-1989 and report errors on 1990-2020)
    :return: Either dataframe or list of errors (MSE and R2) across all given parameters
    """
    # train/test
    if t == 0:
        X_train = X
        X_test = X
        y_train = y
        y_test = y
    else:
        X_train = X[:t]
        y_train = y[:t]
        X_test = X[t:]
        y_test = y[t:]
    # define the variables
    p = X.shape[1]
    beta = cp.Variable(p)
    lam1 = cp.Parameter(nonneg=True)
    lam2 = cp.Parameter(nonneg=True)
    # setup the problem
    problem = cp.Problem(cp.Minimize(objective_fn(X_train, y_train, beta, D, lam1, lam2, alpha)))
    errors = []
    for l1 in lambda_lasso_path:
        for ltv in lambda_tv_path:
            # set regularization parameters
            lam1.value = l1
            lam2.value = ltv
            # solve the objective
            try:
                problem.solve()
                # build prediction
                yhat = X_test@beta.value
                if region is not None:
                    errors.append([region, l1, ltv, r2_score(y_test, yhat), mean_squared_error(y_test, yhat)])
                else:
                    errors.append([l1, ltv, r2_score(y_test, yhat), mean_squared_error(y_test, yhat)])
            except:
                continue
    if return_df:
        if region is not None:
            return pd.DataFrame(errors, columns=['region', 'lambda_1', 'lambda_tv', 'r2', 'mse'])
        else:
            return pd.DataFrame(errors, columns=['lambda_1', 'lambda_tv', 'r2', 'mse'])
    else:
        return errors


def gtv_cv(X, y, D, alpha, lambda_lasso_path, lambda_tv_path):
    # weight data
    n = X.shape[0]
    weights = np.array([alpha ** (n - t) for t in np.arange(1, n + 1)])
    X = X * np.sqrt(weights.reshape(-1, 1))
    y = y * np.sqrt(weights)
    np.random.seed(2) # random seed for reproducibility
    cv = KFold(n_splits=5, shuffle=True)
    for train, test in cv.split(X):
        Xtr = X[train]
        Xtst = X[test]
        ytr = y[train]
        ytst = y[test]
        p = X.shape[1]
        beta = cp.Variable(p)
        lam1 = cp.Parameter(nonneg=True)
        lam2 = cp.Parameter(nonneg=True)
        # setup the problem
        problem = cp.Problem(cp.Minimize(objective_fn(Xtr, ytr, beta, D, lam1, lam2)))
        errors = []
        for l1 in lambda_lasso_path:
            for ltv in lambda_tv_path:
                # set regularization parameters
                lam1.value = l1
                lam2.value = ltv
                # solve the objective
                try:
                    problem.solve()
                    # build prediction
                    yhat = Xtst @ beta.value
                    errors.append([l1, ltv, r2_score(ytst, yhat), mean_squared_error(ytst, yhat)])
                except:
                    continue
    errors = pd.DataFrame(errors, columns=['lambda_1', 'lambda_tv', 'r2', 'mse'])
    (l1, ltv) = errors.groupby(['lambda_1', 'lambda_tv'])[['r2', 'mse']
        ].mean().reset_index().sort_values('mse').iloc[0][['lambda_1', 'lambda_tv']].values
    return errors, l1, ltv

def objective_fn_gtv_only(X, Y, beta, edge_incidence, lam1):
    return loss_fn(X, Y, beta, 1) + lam1 * difference_pen(beta, edge_incidence)

def gtv_only_cvx(X, y, D, lambda_tv):
    """
    Function to actually estimate beta for specific values of regularization parameters
    :param X: n x p matrix of predictors
    :param y: n-dim response vector
    :param D: |E| x p edge-incdience matrix (same as edge_incidence in other functions)
    :param lambda_lasso: Lasso regularizatin parameter
    :param lambda_tv: GTV regularization parameter
    :param alpha: discount penalty between 0 and 1
    :return: estimated p-dim coefficient vector
    """
    # define the variables
    p = X.shape[1]
    beta = cp.Variable(p)
    lam1 = cp.Parameter(nonneg=True)
    # setup the problem
    problem = cp.Problem(cp.Minimize(objective_fn_gtv_only(X, y, beta, D, lam1)))
    # set regularization parameters
    lam1.value = lambda_tv
    # solve the objective
    problem.solve()
    # return coefficients
    return beta.value