#LIBRARIES
import torch
from torchvision import transforms
from tqdm.auto import tqdm
from timeit import default_timer
from torch.optim.lr_scheduler import ReduceLROnPlateau

#SCRIPTS
from Model import DEVICE, getMaskModel
from Utils import accuracy,printTrainTime,getDataLoader,trainStep, saveCheckpoint,loadCheckpoint, validationStep

#HYPERPARAMETERS
EPOCHS = 100
FIRST_STAGE_EPOCHS = 50
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False

#PATHS
TRAIN_IMAGES_DIR = "dataset_cropped\\TrainDatas"
TEST_IMAGES_DIR = "dataset\\TestDatas"
VALIDATION_IMAGES_DIR = "dataset_cropped\\ValDatas"

if __name__ == "__main__":
    
    torch.manual_seed(42)
    
    patience = 25
    patienceCounter = 0
    bestLoss = float("inf")

    trainTransform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),      # ±10 derece rotasyon
        transforms.ColorJitter(             # parlaklık, kontrast, doygunluk
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    testTransform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    validationTransform =  transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    trainDataLoader , testDataLoader, validationDataLoader = getDataLoader(
        trainImagesDir=TRAIN_IMAGES_DIR,
        testImagesDir=TEST_IMAGES_DIR,
        validationImagesDir=VALIDATION_IMAGES_DIR,
        trainTransform=trainTransform,
        testTransform=testTransform,
        validationTransform=validationTransform,
        batchSize=BATCH_SIZE,
        numWorkers=NUM_WORKERS,
        pinMemory=PIN_MEMORY
    )

    #OPTIMIZER / MODEL / LOSS FN
    myModel = getMaskModel(numClasses=29, device=DEVICE)
    lossFn = torch.nn.CrossEntropyLoss()

    print("--- AŞAMA 1 BAŞLIYOR: Sadece son katman (classifier) eğitiliyor ---")

    for param in myModel.parameters():
        param.requires_grad = False

    # Sadece yeni eklediğimiz son katmanı (classifier[1]) aç
    for param in myModel.classifier.parameters():
        param.requires_grad = True


    optimizer = torch.optim.AdamW(params=myModel.classifier.parameters(),
                                  lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                  mode="min",
                                  patience=10)

    startTrainTimer = default_timer()

    if LOAD_MODEL:
        try:
            loadCheckpoint(checkpointFile="myCheckpoint.pth",
                        model=myModel,
                        optimizer=optimizer)
        except FileNotFoundError:
             print("Checkpoint bulunamadı, yeni model eğitiliyor")


    for epoch in tqdm(range(FIRST_STAGE_EPOCHS)):
        print(f"-----EPOCH = {epoch}/{FIRST_STAGE_EPOCHS}")
        trainStep(model=myModel,
                  dataLoader=trainDataLoader,
                  optimizer=optimizer,
                  lossFn=lossFn,
                  accFn=accuracy,
                  device=DEVICE)
        
        # testStep(model=myModel,
        #          dataLoader=testDataLoader,
        #          dataset=trainDataLoader.dataset,
        #          csvFile="test_predictions.csv",
        #          device=DEVICE)

        validationLoss = validationStep(model=myModel,
                                        dataLoader=validationDataLoader,
                                        lossFn=lossFn,
                                        accFn=accuracy,
                                        device=DEVICE)
        
        scheduler.step(validationLoss)

        if validationLoss < bestLoss:
            bestLoss = validationLoss
            patienceCounter = 0
            saveCheckpoint(model=myModel,
                           optimizer=optimizer,
                           epoch=epoch)
        else:
            patienceCounter +=1
            print(f"{patienceCounter} tur boyunca gelişme görülmedi.")
            if patienceCounter == patience:
                print("EARLY STOPPING TRIGGERED")
                break


    # --- AŞAMA 2: FINE-TUNING (Tüm modeli düşük LR ile eğit) ---
    print("\n--- AŞAMA 2 BAŞLIYOR: Tüm model (fine-tuning) eğitiliyor ---")
    
    # Tüm katmanları aç
    for param in myModel.parameters():
        param.requires_grad = True
        
    # Optimizer'ı TÜM parametreler ve DÜŞÜK LR ile yeniden oluştur
    optimizer = torch.optim.AdamW(params=myModel.parameters(),
                                  lr=LEARNING_RATE / 10) # 10 kat daha düşük LR
    
    # Scheduler'ı yeni optimizer ile güncelle
    scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                  mode="min",
                                  patience=10)
    
    # En iyi loss'u ve patience'ı sıfırla (veya en son A1'den devam et)
    # bestLoss = float("inf") # Sıfırlamak daha iyi olabilir
    patienceCounter = 0 
    
    # Kalan epochlar için ana döngü
    remaining_epochs = EPOCHS - FIRST_STAGE_EPOCHS
    for epoch_idx in tqdm(range(remaining_epochs)):
        epoch = epoch_idx + FIRST_STAGE_EPOCHS # Toplam epoch sayısını göstermek için
        print(f"-----AŞAMA 2 - EPOCH = {epoch}/{EPOCHS}-----")
        
        trainStep(model=myModel,
                  dataLoader=trainDataLoader,
                  optimizer=optimizer,
                  lossFn=lossFn,
                  accFn=accuracy,
                  device=DEVICE)
        
        validationLoss = validationStep(model=myModel,
                                        dataLoader=validationDataLoader,
                                        lossFn=lossFn,
                                        accFn=accuracy,
                                        device=DEVICE)
        
        scheduler.step(validationLoss)

        if validationLoss < bestLoss:
            bestLoss = validationLoss
            patienceCounter = 0
            saveCheckpoint(model=myModel,
                           optimizer=optimizer,
                           epoch=epoch)
        else:
            patienceCounter +=1
            print(f"{patienceCounter} tur boyunca gelişme görülmedi.")
            if patienceCounter >= patience:
                print("AŞAMA 2 İÇİN EARLY STOPPING TRIGGERED")
                break 

    endTrainTimer = default_timer()

    printTrainTime(start=startTrainTimer,
                   end=endTrainTimer,
                   device=DEVICE)
    