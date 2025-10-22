#LIBRARIES
import cv2
import torch
import numpy as np
import mediapipe as mp
import torchvision.transforms as transforms
from collections import deque

#SCRIPTS
from Model import getMaskModel, DEVICE
from Dataset import MyDataset

MODEL_PATH = "myCheckpoint.pth"
NUM_CLASSES = 29
IMAGE_SIZE = 224

SMOOTH_WINDOW = 20            # Kararlılık için frame sayısı
CONFIDENCE_THRESHOLD = 0.70   # Minimum güven
MIN_REPEAT_TO_ACCEPT = 6      # Tahminin kabul edilmesi için tekrar
COOLDOWN_FRAMES = 30          # Bekleme süresi

model = getMaskModel(numClasses=NUM_CLASSES, device=DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model"])
model.eval()

try:
    train_dataset = MyDataset(rootDir="dataset_cropped/TrainDatas", tranform=None, labeled=True)
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
except FileNotFoundError:
    train_dataset = MyDataset(rootDir="dataset/TrainDatas", tranform=None, labeled=True)
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def get_square_crop(image, hand_landmarks):
    h, w, _ = image.shape
    x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
    y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    cx, cy = (x_min + x_max)//2, (y_min + y_max)//2
    box_size = max(x_max - x_min, y_max - y_min)
    padding = int(box_size * 0.3)
    box_size = max(box_size + padding * 2, 80)
    x1, y1 = max(0, cx - box_size//2), max(0, cy - box_size//2)
    x2, y2 = min(w, cx + box_size//2), min(h, cy + box_size//2)
    return image[y1:y2, x1:x2], (x1, y1, x2, y2)

def get_stable_prediction(queue):
    if not queue: return None, 0.0
    conf_map = {}
    for cls, conf in queue:
        conf_map.setdefault(cls, []).append(conf)
    scores = {cls: np.mean(v) * len(v) for cls, v in conf_map.items()}
    best_cls = max(scores, key=scores.get)
    return best_cls, np.mean(conf_map[best_cls])

cap = cv2.VideoCapture(0)
prediction_queue = deque(maxlen=SMOOTH_WINDOW)
output_text = ""
prev_class = None
repeat_count = 0
cooldown = 0


while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera okunamadı.")
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if cooldown > 0:
        cooldown -= 1

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            hand_crop, (x1, y1, x2, y2) = get_square_crop(frame_rgb, hand_landmarks)
            if hand_crop.size == 0:
                continue

            try:
                img_tensor = transform(hand_crop).unsqueeze(0).to(DEVICE)
                with torch.inference_mode():
                    logits = model(img_tensor)
                    probs = torch.softmax(logits, dim=1)
                    conf, pred_idx = torch.max(probs, dim=1)
                    conf, pred_idx = conf.item(), pred_idx.item()

                if conf >= CONFIDENCE_THRESHOLD:
                    prediction_queue.append((pred_idx, conf))

                stable_pred, avg_conf = get_stable_prediction(prediction_queue)

                if stable_pred is not None:
                    if prev_class == stable_pred:
                        repeat_count += 1
                    else:
                        repeat_count = 1
                        prev_class = stable_pred

                    # Karar alma
                    if repeat_count >= MIN_REPEAT_TO_ACCEPT and cooldown == 0:
                        pred_name = idx_to_class.get(stable_pred, str(stable_pred)).upper()

                        # ---- DELETE ----
                        if pred_name in ["DELETE", "DEL", "SIL", "BACKSPACE"]:
                            if output_text:
                                output_text = output_text[:-1]
                            cooldown = COOLDOWN_FRAMES

                        # ---- SPACE ----
                        elif pred_name in ["SPACE", "BOSLUK", "BLANK"]:
                            output_text += " "
                            cooldown = int(COOLDOWN_FRAMES * 0.6)  

                        # ---- NORMAL HARF ----
                        else:
                            if len(output_text) >= 1 and output_text[-1] == pred_name:
                                if cooldown == 0:
                                    output_text += pred_name
                                    cooldown = int(COOLDOWN_FRAMES * 1.2)
                            else:
                                output_text += pred_name
                                cooldown = COOLDOWN_FRAMES


                        prediction_queue.clear()
                        repeat_count = 0
                        prev_class = None

            except Exception as e:
                print(f"Hata: {e}")

    cv2.rectangle(frame, (5, 380), (635, 470), (0, 0, 0), -1)
    cv2.putText(frame, output_text, (10, 440),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)

    cv2.imshow("Sign Language Recognition", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('c'):
        output_text = ""
        prediction_queue.clear()
        repeat_count = 0
        prev_class = None
        cooldown = 0

cap.release()
cv2.destroyAllWindows()
hands.close()
