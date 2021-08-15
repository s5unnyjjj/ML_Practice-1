
import numpy as np

random_seed = 7

def cost_naive(X, y, theta):
    n = len(X)
    J = 0
    h = X.dot(theta)

    for i in range(n):
        J += pow(h[i] - y[i], 2)
    J = 1 / (2 * n) * J

    return J


def cost_vectorized(X, y, theta):
    n = len(X)
    h = X.dot(theta)
    J = 1 / (2 * n) * np.sum(pow(h-y, 2))
    return J


def ols_func(X, y):
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return theta


def gradient_descent_func_naive(X, y, theta, alpha, num_iters):
    n = len(X)
    J_his = np.zeros((num_iters,))
    T_his = np.zeros((num_iters, 2))
    for i in range(num_iters):
        T_his[i] = theta
        h = X.dot(theta)
        sum = 0
        for j in range(n):
            sum += np.dot(X[j].T, h[j] - y[j])
        theta = theta - (alpha / n) * sum
        J_his[i] = cost_naive(X, y, theta)
    return theta, J_his, T_his


def gradient_descent_func_vectorized(X, y, theta, alpha, num_iters):
    n = len(X)
    J_his = np.zeros((num_iters,))
    T_his = np.zeros((num_iters, 2))
    for i in range(num_iters):
        T_his[i] = theta
        h = X.dot(theta)
        theta = theta - (alpha/n) * (np.dot(X.T, h-y))
        J_his[i] = cost_vectorized(X, y, theta)
    return theta, J_his, T_his


def stochastic_gradient_descent_func(X, y, theta, alpha, num_iters):
    np.random.seed(random_seed)
    mini_batch = 1
    n = len(X)
    J_his = np.zeros((num_iters,))
    T_his = np.zeros((num_iters, 2))
    for i in range(num_iters):
        T_his[i] = theta
        rand_ind = np.random.randint(0, n)
        h = X.dot(theta)
        value = np.dot(X[rand_ind].T, y[rand_ind]-h[rand_ind])
        theta = theta + (alpha/mini_batch) * value
        J_his[i] = cost_vectorized(X, y, theta)
    return theta, J_his, T_his


