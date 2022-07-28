import numpy as np
import torch
import random
import math
from scipy.optimize import minimize
from scipy.optimize import Bounds


def DeviceInitialization(GPUNum):
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        device = torch.device(GPUNum)
    else:
        device = torch.device('cpu')
    random.seed(2021)
    np.random.seed(2021)
    torch.manual_seed(2021)
    return device


def DataReading(path, task, set, fracTrain, fracTest):
    data = np.genfromtxt(path + task + '\\Csvs\\' + set + '.txt', dtype=str)
    if fracTrain < 100:
        shuffleIdx = np.arange(len(data))
        shuffleRng = np.random.RandomState(2021)
        shuffleRng.shuffle(shuffleIdx)
        data = data[shuffleIdx]
        trainNum = math.ceil(fracTrain * len(data) / 100)
        testNum = math.ceil(fracTest * len(data) / 100)
        testNum = len(data) - testNum
        trainIdx = data[:trainNum]
        testIdx = data[testNum:]
    else:
        trainIdx = data
        testIdx = []

    return trainIdx, testIdx


def costSetting(x):
    global cost
    cost = x


def balanceSetting(x):
    global balance
    balance = x


def obj(w):
    return (w*cost).sum() + balance*(w*w).sum()


def constraint(w):
    return w.sum() - 1


def weightOptim(cost, balance):
    costSetting(cost)
    balanceSetting(balance)
    w0 = np.ones_like(cost) / cost.shape[0]
    bound = Bounds(np.zeros_like(cost), np.ones_like(cost))
    weight = minimize(obj, w0, method='SLSQP', tol=1e-6, bounds=bound, constraints={'fun': constraint, 'type': 'eq'})
    weight = weight.x

    return weight*len(weight)