#LIBRARIES
import torch
from torchvision import transforms
from tqdm.auto import tqdm
from timeit import default_timer

#SCRIPTS
from Model import DEVICE, pretrainedModel
from Utils import accuracy,printTrainTime,getDataLoader,trainStep,testStep, saveCheckpoint,loadCheckpoint

#HYPERPARAMETERS
EPOCHS = 1
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False

#PATHS
TRAIN_IMAGES_DIR = "dataset\\TrainDatas"
TEST_IMAGES_DIR = "dataset\\TestDatas"

if __name__ == "__main__":
    torch.manual_seed(42)
    
    patience = 25
    patienceCounter = 0
    bestLoss = float("inf")

    trainTransform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])

    testTransform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])

    trainDataLoader , testDataLoader = getDataLoader(
        trainImagesDir=TRAIN_IMAGES_DIR,
        testImagesDir=TEST_IMAGES_DIR,
        trainTransform=trainTransform,
        testTransform=testTransform,
        batchSize=BATCH_SIZE,
        numWorkers=NUM_WORKERS,
        pinMemory=PIN_MEMORY
    )

    #OPTIMIZER / MODEL / LOSS FN
    myModel = pretrainedModel
    optimizer = torch.optim.AdamW(params=myModel.parameters(),
                                  lr=LEARNING_RATE)
    lossFn = torch.nn.CrossEntropyLoss()

    startTrainTimer = default_timer()

    if LOAD_MODEL:
        try:
            loadCheckpoint(checkpointFile="myCheckpoint.pth",
                        model=myModel,
                        optimizer=optimizer)
        except FileNotFoundError:
             print("Checkpoint bulunamadı, yeni model eğitiliyor")


    for epochs in tqdm(range(EPOCHS)):
        trainStep(model=myModel,
                  dataLoader=trainDataLoader,
                  optimizer=optimizer,
                  lossFn=lossFn,
                  accFn=accuracy,
                  device=DEVICE)
        
        testStep(model=myModel,
                 dataLoader=testDataLoader,
                 lossFn=lossFn,
                 accFn=accuracy,
                 device=DEVICE)
    

    endTrainTimer = default_timer()

    printTrainTime(start=startTrainTimer,
                   end=endTrainTimer,
                   device=DEVICE)
    

