#LIBRARIES
import torch
import csv
from torch.utils.data import DataLoader, random_split, Subset

#SCRIPTS
from Model import DEVICE
from Dataset import MyDataset

def saveCheckpoint(model,optimizer,epoch, checkpointFile = "myCheckpoint.pth"):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }

    torch.save(checkpoint, checkpointFile)
    print("MODEL CHECKPOINT ALINDI")

def loadCheckpoint(checkpointFile,model,optimizer):
    checkpoint = torch.load(checkpointFile, map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    epoch = checkpoint["epoch"]
    print("CHECKPOINT YUKLENDİ")
    return epoch + 1


def accuracy(yTrue, yPred):
    preds = torch.argmax(yPred, dim=1) 
    correct =  torch.eq(preds, yTrue).sum().item()
    acc = correct / len(yTrue)
    return acc

def printTrainTime(start,end,device):
    totalTime = end - start
    print(f"Total time is {totalTime} on the {device}")

def getDataLoader(trainImagesDir,
                  testImagesDir,
                  validationImagesDir,
                  trainTransform,
                  testTransform,
                  validationTransform,
                  batchSize,
                  numWorkers,
                  pinMemory):
    
    trainDatas = MyDataset(
        rootDir=trainImagesDir,
        tranform=trainTransform,
        labeled=True
    )

    testDatas = MyDataset(
        rootDir=testImagesDir,
        tranform=testTransform,
        labeled=False
    )

    validationDatas = MyDataset(
        rootDir=validationImagesDir,
        tranform=validationTransform,
        labeled=True
    )

    trainDataLoader = DataLoader(
        dataset=trainDatas,
        batch_size=batchSize,
        shuffle=True,
        num_workers=numWorkers,
        pin_memory=pinMemory
    )

    testDataLoader = DataLoader(
        dataset=testDatas,
        batch_size=batchSize,
        shuffle=False,
        num_workers=numWorkers,
        pin_memory=pinMemory
    )

    validationDataLoader = DataLoader(
        dataset=validationDatas,
        batch_size=batchSize,
        shuffle=False,
        num_workers=numWorkers,
        pin_memory=pinMemory
    )

    return trainDataLoader, testDataLoader, validationDataLoader


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
            trainAcc += accFn(yTrue = yTrain, yPred = trainPred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    trainLoss /= len(dataLoader)
    trainAcc /= len(dataLoader)
    print(f"TRAIN LOSS = {trainLoss:.4f} | TRAIN ACCURACY = {trainAcc:.4f}")

# def testStep(model: torch.nn.Module,
#              dataLoader: torch.utils.data.DataLoader,
#              dataset, 
#              csvFile: str = "test_predictions.csv",
#              device: torch.device = DEVICE):

#     model.eval()
#     results = []

#     base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset

#     if not hasattr(base_dataset, "class_to_idx"):
#         raise ValueError("Base dataset does not have class_to_idx mapping.")

#     idx_to_class = {v: k for k, v in base_dataset.class_to_idx.items()}

#     with torch.inference_mode():
#         for xTest, names in dataLoader:
#             xTest = xTest.to(device)
#             yPred = model(xTest)
#             preds = torch.argmax(yPred, dim=1)

#             for name, p in zip(names, preds):
#                 results.append((name, idx_to_class[p.item()]))

#     # CSV'ye yaz (isim sırasını bozmadan)
#     results.sort(key=lambda x: x[0])  # opsiyonel: alfabetik sıralama
#     with open(csvFile, mode="w", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
#         writer.writerow(["image_name", "predicted_class"])
#         for name, pred in results:
#             writer.writerow([name, pred])

#     print(f"Test tahminleri {csvFile} dosyasına kaydedildi.")


def validationStep(model:torch.nn.Module,
                   dataLoader: torch.utils.data.DataLoader,
                   lossFn: torch.nn.Module,
                   accFn,
                   device: torch.device = DEVICE):
    model.eval()
    validationLoss, validationAcc = 0,0
    with torch.inference_mode(): 
        for xVal, yVal in dataLoader:
            xVal, yVal = xVal.to(device), yVal.to(device)

            yPred = model(xVal)
            loss = lossFn(yPred, yVal)
            validationLoss += loss.item()
            validationAcc += accFn(yTrue=yVal, yPred=yPred)

    validationLoss /= len(dataLoader)
    validationAcc /= len(dataLoader)

    print(f"VALIDATION LOSS = {validationLoss:.4f} | VALIDATION ACC = {validationAcc:.4f}")
    return validationLoss




        