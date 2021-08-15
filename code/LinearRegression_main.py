
import time
from LinearRegression_utils import *
from LinearRegression_model import *


def func_linear(x, y):
    print('\n### Check ###')
    # Initialize thetas
    theta = np.zeros(2)

    n_cost = cost_naive(x, y, theta)
    print('cost by naive : %.2f' % n_cost)
    v_cost = cost_vectorized(x, y, theta)
    print('cost by vectorized : %.2f' % v_cost)
    print('cost difference : %f' % np.abs(n_cost - v_cost))


def func_ols(x, y, train_data):
    print('\n### Check ###')
    ols_theta = ols_func(x, y)
    print('Theta from OLS function')
    print('OLS theta 0 :', ols_theta[0])
    print('OLS theta 1 :', ols_theta[1])

    plotOLS(train_data, ols_theta)

    return ols_theta

def func_visual(x, y):
    theta_0, theta_1 = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-1, 4, 100))

    J_vals = np.zeros(theta_0.shape)

    for i in range(theta_0.shape[0]):
        for j in range(theta_0.shape[1]):
            J_vals[i, j] = cost_vectorized(x, y, np.array([theta_0[i, j], theta_1[i, j]]))

    plotCostDist(theta_0, theta_1, J_vals)

    return J_vals

def func_batchGD(x, y, ols, j_val, train_data):
    iterations = 3000
    alpha = 0.01

    theta = np.zeros(2)
    theta_0, theta_1 = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-1, 4, 100))
    print('\n### Check ###')
    s_time = time.time()
    theta_n, J_his_n, T_his_n = gradient_descent_func_naive(x, y, theta, alpha, iterations)
    print('theta by naive : {} with {:.3f}s'.format(theta_n, time.time() - s_time))
    s_time = time.time()
    theta_v, J_his_v, T_his_v = gradient_descent_func_vectorized(x, y, theta, alpha, iterations)
    print('theta by vectorized : {} with {:.3f}s'.format(theta_v, time.time() - s_time))

    plotCostOpt(J_his_v)

    plotTraj(theta_0, theta_1, ols, j_val, T_his_v)

    " Check how the line by GD changes as the number of updates (iteration) increases "
    plotIters(500, train_data, ols, T_his_v)

    return theta_v, T_his_v


def func_stochasticGD(x, y, ols, j_val, T_his_v1):
    iterations = 3000
    alpha = 0.01

    theta = np.zeros(2)
    theta_0, theta_1 = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-1, 4, 100))
    print('\n### Check ###')
    SGDtheta_n, SGDJ_his_n, SGDT_his_n = stochastic_gradient_descent_func(x, y, theta, alpha, iterations)
    print('theta by SGD : {} '.format(SGDtheta_n))

    plotTrajCompare(theta_0, theta_1, ols, j_val, T_his_v1, SGDT_his_n)


def predict(train_data, theta_v, x, y, ols):
    plotPred(train_data, theta_v, [7.5, 15.])

    testData = pandas.read_csv('data_test.txt', header=None, names=['population', 'profit'])

    testData.insert(0, 'ones', 1.)

    testX = testData[['ones', 'population']].values
    testY = testData['profit'].values

    # Training MSE
    print('\n### Check ###')
    prediction = np.dot(x, ols)
    mse = np.mean((prediction - y) ** 2)
    print('OLS training_mse=', mse)

    prediction = np.dot(x, theta_v)
    mse = np.mean((prediction - y) ** 2)
    print('Gradient based training_mse=', mse)

    # Test MSE
    print('\n### Check : Test MSE between OLS and Gradient')
    prediction = np.dot(testX, ols)
    predict_mse = np.mean((prediction - testY) ** 2)
    print('OLS test_mse=', predict_mse)

    prediction = np.dot(testX, theta_v)
    predict_mse = np.mean((prediction - testY) ** 2)
    print('Gradient based test_mse=', predict_mse)


if __name__ == "__main__":
    trainData = pandas.read_csv('data_train.txt', header=None, names=['population', 'profit'])

    plotData1_1(trainData)

    trainX = trainData['population'].values
    trainY = trainData['profit'].values

    trainX = np.expand_dims(trainX, 1)
    trainX = np.insert(trainX, 0, np.ones(len(trainX)), 1)

    func_linear(trainX, trainY)

    thetaOLS = func_ols(trainX, trainY, trainData)

    valJ = func_visual(trainX, trainY)

    thetaV, T_hisV = func_batchGD(trainX, trainY, thetaOLS, valJ, trainData)

    func_batchGD(trainX, trainY, thetaOLS, valJ, trainData)

    func_stochasticGD(trainX, trainY, thetaOLS, valJ, T_hisV)

    predict(trainData, thetaV, trainX, trainY, thetaOLS)

