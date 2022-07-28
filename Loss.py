import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self, mode):
        super(Loss, self).__init__()
        self.mode = mode

    def forward(self, output, target):
        if self.mode == 'Dice':
            metric = DSC()
        elif self.mode == 'CE':
            metric = CE()
        else:
            print('The chosen loss function is not provided!!!! We use Dice Loss instead!!!')
            metric = DSC()

        batchSize = target.size(0)
        classNum = target.size(1)
        loss = torch.empty(batchSize, classNum)
        for ii in range(classNum):
            loss[:, ii] = metric(output[:, ii], target[:, ii])
        return loss


class DSC(nn.Module):
    def __init__(self):
        super(DSC, self).__init__()

    def forward(self, output, target):
        batchSize = target.size(0)
        target = target.view(batchSize, -1)
        output = output.view(batchSize, -1)
        inter = (output*target).sum(1)
        union = output.sum(1) + target.sum(1) + 1e-10
        loss = 1 - (2. * inter + 1e-10) / union
        return loss


class CE(nn.Module):
    def __init__(self):
        super(CE, self).__init__()

    def forward(self, output, target):
        outputF = torch.log(output + 1e-10)
        outputB = torch.log(1 - output + 1e-10)
        loss = -target * outputF - (1 - target) * outputB
        return loss.mean()