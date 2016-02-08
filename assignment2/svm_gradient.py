import numpy as np
import sklearn.preprocessing as pp
import sklearn.linear_model as lm
import pdb
import pandas
from matplotlib import pyplot


class LostVisualiser(object):
    def __init__(self, x_range=1000, y_range=1000):
        self.data_points = []
        self.series = []
        self.x_range = x_range
        self.y_range = y_range

    def start(self):
        self.data_points = []
        self.series = []
        pyplot.axis([0, self.x_range, 0, self.y_range])
        pyplot.ion()

    def update(self, data):
        self.data_points.append(data)
        self.series.append(len(self.series)+1)
        pyplot.plot(self.data_points, self.series)
        pyplot.draw()


def get_dataset(name):
    data = pandas.read_csv(name,
                           na_values=["?"],
                           header=None,
                           skipinitialspace=True).dropna()
    data = data.reindex(np.random.permutation(data.index))
    dataX = data.select_dtypes([np.number])
    # for data<50k -> 1 and data>50k -> 0
    dataY = (data[len(data.columns)-1] == "<=50K")*2-1
    return dataX.as_matrix(), dataY.as_matrix()


# svm error function E = max((1-y(ax-b))^2, 0)+lambda/2*a^2
# a(n+1) = a(n) - eta(lamda_a)-(y(ax+b)>1)
# b(n+1) = b(n) (y(ax+b)<1)

def accuracy(testX, testY, a, b, scaler=None):
    testX = scaler.transform(testX) if scaler is not None else testX
    return sum(testY*(testX.dot(a)+b) > 0)/len(testY)


def hinge_loss(testX, testY, a, b, l):
    """
    @type testX: np.array
    @type testY: np.array
    @type a: np.array
    """
    return np.average(np.maximum((1-testY*(testX.dot(a)+b)), 0))+l/2*a.dot(a)


# def update(a, b, x, y, e, l):
#     errors = (y*(x.dot(a)+b) < 1)
#     a -= e*(len(errors)*l*a-(y*errors).dot(x))
#     b -= e*np.dot(y, errors)

def update(a, b, X, Y, e, l):
    a = a.copy()
    for i in range(len(Y)):
        x = X[i]
        y = Y[i]
        if y*(a.dot(x)+b) >= 1:
            a = (1-e*l)*a
            b = b
        else:
            a = a-e*(l*a-y*x)
            b = b+e*y
    return a, b


def train(trainX, trainY, iters=50, l=1, interval=300,
          plotter=None, testX=None, testY=None,
          epoch_fun=lambda x: 1/(0.01*x+10)):

    (m, n) = trainX.shape
    a = np.zeros(n)
    b = 0

    plotter.start() if plotter else None
    # testX = testX or trainX
    # testY = testY or trainY

    loss = []
    acc = []
    for iter in range(iters):
        rands = np.random.randint(0, m, interval)
        e = epoch_fun(iter)
        x = trainX[rands]
        y = trainY[rands]
        a, b = update(a, b, x, y, e, l)
        loss.append(hinge_loss(trainX, trainY, a, b, l))
        acc.append(accuracy(testX, testY, a, b))
    return (a, b, loss, acc)


def predict(testX, a, b, scaler=None):
    testX = scaler.transform(testX) if scaler is not None else testX
    return np.sign(testX.dot(a)+b)


def kernel(trainX):
    n, m = trainX.shape
    kernelX = np.zeros((n, (m+1)*m/2))
    c = 0
    for i in range(m):
        for j in range(i+1):
            kernelX[:, c] = trainX[:, i]*trainX[:, j]
            c = c+1
    return kernelX


def perform(l=0.1, interval=300, iters=50):
    dataX, dataY = get_dataset('./adult.data')
    n, m = dataX.shape
    trainX = dataX[:int(m*.9), :]
    trainY = dataY[:int(m*.9)]
    scaler = pp.StandardScaler().fit(dataX)
    testX = dataX[int(m*.9):, :]
    testY = dataY[int(m*.9):]
    testX, testY = get_dataset('./adult.test')
    (a, b, lost, acc) = train(scaler.transform(trainX), trainY,
                              testX=scaler.transform(testX), testY=testY,
                              l=l, interval=interval, iters=iters)
    return (a, b), (lost, acc), scaler

if __name__ == "__main__":
    perform()
