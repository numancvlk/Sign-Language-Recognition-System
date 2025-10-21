# LIBRARIES
import cv2
import torch
import numpy as np
import mediapipe as mp
import torchvision.transforms as transforms 

#SCRIPTS
from Model import getMaskModel, DEVICE
from Dataset import MyDataset

# HYPERPARAMETERS
MODEL_PATH = "myCheckpoint.pth" 
NUM_CLASSES = 29
IMAGE_SIZE = 224

model = getMaskModel(numClasses=NUM_CLASSES, device=DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint["model"])
model.eval() 

try:
    train_dataset = MyDataset(rootDir="dataset_cropped/TrainDatas", tranform=None, labeled=True)
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
    print("Class haritası 'dataset_cropped/TrainDatas' üzerinden başarıyla yüklendi.")
except FileNotFoundError:
    print("HATA: 'dataset_cropped/TrainDatas' bulunamadı. Lütfen önce preprocess.py script'ini çalıştırın.")
    print("Geçici olarak 'dataset/TrainDatas' kullanılacak, ancak bu HATALI tahminlere yol açabilir!")
    train_dataset = MyDataset(rootDir="dataset/TrainDatas", tranform=None, labeled=True)
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
except Exception as e:
    print(f"Class haritası yüklenirken kritik hata: {e}")
    exit()


mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def get_square_crop(image, hand_landmarks):
    h, w, c = image.shape
    x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
    y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    box_size = max(x_max - x_min, y_max - y_min)
    
    padding = int(box_size * 0.2)
    box_size = box_size + padding * 2

    x1 = max(0, center_x - box_size // 2)
    y1 = max(0, center_y - box_size // 2)
    x2 = min(w, center_x + box_size // 2)
    y2 = min(h, center_y + box_size // 2)
    
    return image[y1:y2, x1:x2], (x1, y1, x2, y2)


cap = cv2.VideoCapture(0) 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb) 

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            hand_crop, (x1, y1, x2, y2) = get_square_crop(frame_rgb, hand_landmarks)

            if hand_crop.size > 0:
                try:
                    input_tensor = transform(hand_crop).unsqueeze(0).to(DEVICE)
                    with torch.inference_mode():
                        output = model(input_tensor)
                        probabilities = torch.softmax(output, dim=1)
                        confidence = torch.max(probabilities).item()
                        pred_idx = torch.argmax(probabilities, dim=1).item()
                        
                        pred_class = idx_to_class[pred_idx]
                        
                        # Ekrana yazdırma
                        text = f"{pred_class} ({confidence:.2f})"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                except Exception as e:
                    print(e)
                    pass

    cv2.imshow("ASL Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()