import numpy
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


def get_dataset():
    data = pandas.read_csv('./adult.data',
                           na_values=["?"],
                           header=None,
                           skipinitialspace=True).dropna()
    dataX = data.select_dtypes([numpy.number])
    # for data<50k -> 1 and data>50k -> 0
    dataY = (data[len(data.columns)-1] == "<=50K")*2-1
    return dataX.as_matrix(), dataY.as_matrix()


# svm error function E = max((1-y(ax-b))^2, 0)+lambda/2*a^2
# a(n+1) = a(n) - eta(lamda_a)-(y(ax+b)>1)
# b(n+1) = b(n) (y(ax+b)<1)


def hinge_lost(testX, testY, a, b):
    """
    @type testX: numpy.array
    @type testY: numpy.array
    @type a: numpy.array
    """
    return numpy.max(testY*(1-testX.dot(a)+b), 0)


def update(a, b, x, y, e, l):
    # pdb.set_trace()
    errors = (y*(x.dot(a)+b) < 1)
    a -= e*(len(errors)*l*a-(y*errors).dot(x))
    b -= e*numpy.dot(y, errors)


def train(trainX, trainY, iters=1000, l=1, interval=10,
          plotter=None, testX=None, testY=None):
    (m, n) = trainX.shape
    a = numpy.zeros(n)
    b = 0

    plotter.start() if plotter else None
    testX = trainX if not testX else testX
    testY = trainY if not testY else testY

    for iter in range(iters):
        rands = numpy.random.randint(0, m, interval)
        e = 1/(0.01*iter+50)
        x = trainX[rands]
        y = trainY[rands]
        update(a, b, x, y, e, l)
        plotter.update(hinge_lost(testX, testY)) if plotter else None
        print(hinge_lost(testX, testY, a, b))
    return (a, b)


def predict(testX, a, b):
    return numpy.sign(testX.dot(a)+b)


if __name__ == "__main__":
    dataX, dataY = get_dataset()
    trainX = None
    trainY = None
    testX = None
    testY = None
    (a, b) = train(x, y)
