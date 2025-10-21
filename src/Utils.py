#LIBRARIES
import torch

#SCRIPTS
from Model import DEVICE

def accuracy(yTrue, yPred):
    correct =  torch.eq(yTrue,yPred).sum().item()
    acc = correct / len(yTrue)
    return acc

def printTrainTime(start,end,device):
    totalTime = end - start
    print(f"Total time is {totalTime} on the {device}")

def trainStep(model: torch.nn.Module,
              dataLoader: torch.utils.data.DataLoader,
              optimizer: torch.optim.Optimizer,
              lossFn: torch.nn.Module,
              accFn,
              device: torch.device = DEVICE):
    
    model.train()
    trainLoss, trainAcc = 0,0

    for batch, (xTrain,yTrain) in enumerate(dataLoader):
        xTrain, yTrain = xTrain.to(device), yTrain.to(device)

        with torch.autocast(device_type="cuda"):
            trainPred = model(xTrain)
            loss = lossFn(trainPred,yTrain)
            trainLoss += loss.item()
            trainAcc += accFn(yTrue = yTrain, yPred = trainPred.argmax(dim=1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    trainLoss /= len(dataLoader)
    trainAcc /= len(dataLoader)
    print(f"TRAIN LOSS = {trainLoss} | TRAIN ACCURACY = {trainAcc}")

def testStep(model: torch.nn.Module,
             dataLoader: torch.utils.data.DataLoader,
             lossFn: torch.nn.Module,
             accFn,
             device: torch.device = DEVICE):
    
    model.eval()
    testLoss, testAcc = 0,0

    for xTest, yTest in dataLoader:
        xTest, yTest = xTest.to(device), yTest.to(device)

        with torch.inference_mode():
            testPred = model(xTest)
            testLoss += lossFn(testPred,yTest).item()
            testAcc += accFn(yTrue = yTest, yPred = testPred.argmax(dim=1))
        
    testLoss /= len(dataLoader)
    testAcc /= len(dataLoader)

    print(f"TEST LOSS = {testLoss} | TEST ACCURACY = {testAcc}")



        