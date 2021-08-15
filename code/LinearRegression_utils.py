
import matplotlib.pyplot as plt
import numpy as np
import pandas


def plotData1_1(data):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(data.population, data.profit)
    ax.set_xlabel('population', size=14)
    ax.set_ylabel('profit', size=14)
    ax.set_title('Distribution of the Profit and the Population', fontsize=15)
    plt.setp(ax.get_xticklabels(), fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)


def plotOLS(data, theta):
    plotData1_1(data)
    X = data.population
    x_plot = np.array([np.min(X), np.max(X)])
    y_plot = theta[1] * x_plot + theta[0]
    plt_1 = plt.plot(x_plot, y_plot, 'r-', label='y = ${:.2f} x + ({:.2f})$'.format(theta[1], theta[0]))
    plt.title('Linear regression by OLS theta', fontsize=15)
    plt.legend()
    plt.show()


def plotCostDist(theta_0, theta_1, cost_values):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(theta_0, theta_1, cost_values)

    ax.set_xlabel('$\\theta_0$', size=15)
    ax.set_ylabel('$\\theta_1$', size=15)
    ax.set_title('Contour of a loss function', fontsize=20, fontweight='bold')
    plt.show()


def plotCostOpt(cost_history):
    pandas.Series(cost_history).plot(title='Learning curve', fontsize=15, figsize=(8, 5))
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Cost', fontsize=14)
    plt.show()


def plotPred(data, theta, test_data):
    plotData1_1(data)
    #     data.plot(x=x, y=y, kind='scatter')
    X_test = np.array([[1, test_data[0]], [1, test_data[1]]])
    y_pred = X_test @ theta
    plt.plot(X_test[:, 1], y_pred, 'ro')
    plt.title('Profit prediction', fontsize=15)
    plt.show()


def plotTraj(theta_0, theta_1, opt_theta, cost_values, theta_history):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.contour(theta_0, theta_1, cost_values, levels=np.logspace(-2, 3, 15))
    ax.plot(opt_theta[0], opt_theta[1], 'b+', markersize=10, label="OLS theta")
    index = np.arange(10) * 150
    ax.plot(theta_history[index, 0], theta_history[index, 1], 'r.-', label='Trajectory of gradient descent thetas')
    ax.set_xlabel('$\\theta_0$', size=15)
    ax.set_ylabel('$\\theta_1$', size=15)
    ax.set_title('Theta convergence trajectory', fontsize=15, fontweight='bold')
    ax.legend()
    plt.show()

def plotIters(split, data, ols_theta, theta_history1):
    interval = int(len(theta_history1) / split)
    idxx = (np.arange(interval)) * split
    theta_history1 = theta_history1[idxx]

    fig = plt.figure(figsize=(25, 20))
    for i, t in enumerate(theta_history1[:interval]):
        ax = fig.add_subplot(interval, 3, i + 1)
        Xs = np.linspace(0, 25, 100)
        Ygd = t[0] + Xs * t[1]
        Yols = ols_theta[0] + Xs * ols_theta[1]
        ax.autoscale(tight=True)
        #       ax.set_ylim(-10,30)
        ax.scatter(data.population, data.profit)
        ax.plot(Xs, Yols, 'b')
        ax.plot(Xs, Ygd, 'r')
        plt.legend(["OLS", "Gradient descent"])
        ax.set_title('\n%d th iteration' % idxx[i], fontsize=15)
    fig.tight_layout()


def plotTrajCompare(theta_0, theta_1, opt_theta, cost_values, theta_history1, theta_history2):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.contour(theta_0, theta_1, cost_values, levels=np.logspace(-2, 3, 15))
    ax.plot(opt_theta[0], opt_theta[1], 'b+', markersize=10, label="OLS theta")
    index = np.arange(10) * 150
    ax.plot(theta_history1[index, 0], theta_history1[index, 1], 'r.-', label='Trajectory of gradient descent thetas')
    ax.plot(theta_history2[index, 0], theta_history2[index, 1], 'g.-',
            label='Trajectory of stochastic gradient descent thetas')
    ax.set_xlabel('$\\theta_0$', size=15)
    ax.set_ylabel('$\\theta_1$', size=15)
    ax.set_title('Theta convergence trajectory', fontsize=15, fontweight='bold')
    ax.legend()
    plt.show()


