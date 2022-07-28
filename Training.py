import torch
import time
from Tools import DeviceInitialization
from Tools import DataReading
import numpy as np
import math
from Dataset import DataSet
from torch.utils.data import DataLoader
from itertools import cycle
from Model import UNet
from Model import ResNet
from Model import Inception
from Loss import Loss
from Tools import weightOptim


task = 'CT' ## 'CT' or 'Cell'
set = 'BTCV' ## 'BTCV' or 'C5'
set1 = 'MSD' ## 'MSD' or 'ISBI'
set2 = 'TCIA' ## 'TCIA' or 'CCE'
batch_size = 8
class_num = 8 ## 8 or 3
epoch_num = 40
learning_rate = 3e-4
weightReg = 2.0
path = 'D:\\DataDiscernment\\Data\\'
modelPath = 'D:\\DataDiscernment\\Model\\Our\\'
fracTrain = 20
fracTest = 80
fracExtra = 100
####################################################
netName = '_UNet_' ##UNet, ResNet or Inception
lossType = 'Dice' ##'Dice' or 'CE'
Net = UNet(obj_num=class_num)    ########
ConsNet = UNet(obj_num=class_num)
#Net = ResNet(obj_num=class_num)  #########
#ConsNet = ResNet(obj_num=class_num)
#Net = Inception(obj_num=class_num)  ######
#ConsNet = Inception(obj_num=class_num)
modelName = task + netName + lossType + '_'
####################################################
device = DeviceInitialization('cuda:0')
trainIdx, testIdx = DataReading(path=path, task=task, set=set, fracTrain=fracTrain, fracTest=fracTest)
trainIdx1, _ = DataReading(path=path, task=task, set=set1, fracTrain=fracExtra, fracTest=0)
trainIdx2, _ = DataReading(path=path, task=task, set=set2, fracTrain=fracExtra, fracTest=0)

flagTrain = list(np.zeros(len(trainIdx)))
flag1 = list(np.ones(len(trainIdx1)))
flag2 = list(np.ones(len(trainIdx2)))
trainIdxAll = np.concatenate((trainIdx, trainIdx1, trainIdx2), axis=0)
flagAll = np.concatenate((flagTrain, flag1, flag2), axis=0)
InterSize = len(trainIdx)
tempLossInternal = torch.ones(InterSize)
tempLossExternal = torch.ones(InterSize+len(trainIdx1)+len(trainIdx2))

internalSet = DataSet(dataPath=path, task=task, dataName=trainIdx, height=128, width=128, flag=False)
trainSet = DataSet(dataPath=path, task=task, dataName=trainIdxAll, height=128, width=128, flag=flagAll)
testSet = DataSet(dataPath=path, task=task, dataName=testIdx, height=128, width=128, flag=False)
InternalSet = torch.utils.data.DataLoader(dataset=internalSet, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
InternalSet = cycle(InternalSet)
TrainSet = torch.utils.data.DataLoader(dataset=trainSet, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
TestSet = torch.utils.data.DataLoader(dataset=testSet, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
Net.to(device)

optim = torch.optim.Adam(Net.parameters(), lr=learning_rate)
criterion = Loss(mode=lossType)
consName = 'D:\\DataDiscernment\\Model\\Baseline\\' + task +netName + lossType + '_1.pkl'
ConsNet.load_state_dict(torch.load(consName))
ConsNet.to(device)
ConsNet.eval()
xi = 0.05
xiRate = 1e-4
xiWeight = 1.0
start_time = time.time()
IterNum = 0
idxInter = 0
for epoch in range(epoch_num):
    Net.train()
    tempLoss = 0
    if epoch > 30:
        xiWeight = 0.5*(1 + math.cos(0.1 * math.pi * (epoch - 30)))
    if epoch < 10:
        for idx, (images, targets, _) in enumerate(TrainSet, 0):
            images = images.to(device)
            targets = targets.to(device)
            sampleNum = images.size(0)
            outputs = Net(images)
            loss = criterion(outputs, targets).mean(dim=1)
            tempLossExternal[idx * batch_size:idx * batch_size + sampleNum] = loss.detach()
            loss = loss.mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            tempLoss += loss
        IterNum += (idx + 1)
        print("Epoch: %02d  ||  Iteration: %04d  ||  Loss: %.4f  ||  Time elapsed: %.2f(min)"
              % (epoch + 1, IterNum, tempLoss / (idx + 1), (time.time() - start_time) / 60))
    elif (epoch >= 10) & (epoch < 20):
        for idx, (images, targets, flag) in enumerate(TrainSet, 0):
            images = images.to(device)
            targets = targets.to(device)
            outputs = Net(images)
            loss = criterion(outputs, targets).mean(dim=1)
            weight = torch.ones_like(loss)
            if flag.sum() > 0:
                imagesVal, targetsVal, _ = next(InternalSet)
                imagesVal = imagesVal.to(device)
                targetsVal = targetsVal.to(device)
                numInter = imagesVal.size(0)
                sampleNum = images.size(0)
                with torch.no_grad():
                    outputsVal = Net(imagesVal)
                lossVal = criterion(outputsVal, targetsVal).mean(dim=1)
                tempValMean = tempLossInternal.mean() - (tempLossInternal[idxInter * batch_size:idxInter * batch_size + numInter] - lossVal).mean()
                tempExter = tempLossExternal[idx * batch_size: idx * batch_size + sampleNum] - loss.detach()
                cost = abs((loss.detach() - tempValMean) * tempExter)
                costReal = cost[flag > 0]
                tempLossInternal[idxInter * batch_size: idxInter * batch_size + numInter] = lossVal
                tempLossExternal[idx * batch_size: idx * batch_size + sampleNum] = loss.detach()
                if idxInter * batch_size + numInter > InterSize - 1:
                    idxInter = 0
                else:
                    idxInter += 1
                weightTemp = weightOptim(cost=costReal.numpy(), balance=weightReg)
                weight[flag > 0] = torch.from_numpy(weightTemp).type(torch.float)
            loss = (loss * weight).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            tempLoss += loss
        IterNum += (idx + 1)
        print("Epoch: %02d  ||  Iteration: %04d  ||  Loss: %.4f  ||  Time elapsed: %.2f(min)"
              % (epoch + 1, IterNum, tempLoss / (idx + 1), (time.time() - start_time) / 60))
    else:
        for idx, (images, targets, flag) in enumerate(TrainSet, 0):
            images = images.to(device)
            targets = targets.to(device)
            outputs = Net(images)
            loss = criterion(outputs, targets).mean(dim=1)
            weight = torch.ones_like(loss)
            if flag.sum() > 0:
                imagesVal, targetsVal, _ = next(InternalSet)
                imagesVal = imagesVal.to(device)
                targetsVal = targetsVal.to(device)
                numInter = imagesVal.size(0)
                sampleNum = images.size(0)
                with torch.no_grad():
                    outputsVal = Net(imagesVal)
                lossVal = criterion(outputsVal, targetsVal).mean(dim=1)
                tempValMean = tempLossInternal.mean() - (tempLossInternal[idxInter * batch_size:idxInter * batch_size + numInter] - lossVal).mean()
                tempExter = tempLossExternal[idx * batch_size: idx * batch_size + sampleNum] - loss.detach()
                cost = abs((loss.detach() - tempValMean) * tempExter)
                costReal = cost[flag > 0]
                tempLossInternal[idxInter * batch_size: idxInter * batch_size + numInter] = lossVal
                tempLossExternal[idx * batch_size: idx * batch_size + sampleNum] = loss.detach()
                if idxInter * batch_size + numInter > InterSize - 1:
                    idxInter = 0
                else:
                    idxInter += 1
                weightTemp = weightOptim(cost=costReal.numpy(), balance=weightReg)
                weight[flag > 0] = torch.from_numpy(weightTemp).type(torch.float)

            if flag.sum() < images.size(0):
                with torch.no_grad():
                    outputCons = ConsNet(images)
                lossCon = criterion(outputCons, targets).mean(dim=1)
                lossCon = (loss - lossCon) * (1 - flag)
                lossCon = lossCon[lossCon > 0]
                if len(lossCon) > 0:
                    lossCon = lossCon.mean()
                    xiUpdate = lossCon.detach().item()
                else:
                    lossCon = 0
                    xiUpdate = 0
            else:
                lossCon = 0
                xiUpdate = 0
            loss = (weight * loss).mean() + xiWeight * xi * lossCon
            optim.zero_grad()
            loss.backward()
            optim.step()
            xi += xiRate * xiUpdate
            tempLoss += loss
        IterNum += (idx + 1)
        print("Epoch: %02d  ||  Iteration: %04d  ||  Loss: %.4f  ||  Time elapsed: %.2f(min)"
              % (epoch + 1, IterNum, tempLoss / (idx + 1), (time.time() - start_time) / 60))

torch.save(Net.state_dict(), modelPath + modelName + '1.pkl')