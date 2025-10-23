# Sign-Language-Recognition-System
# [TR]
## Projenin Amacı
Bu projede, **MobileNetV2** modelini kullanarak kameradan alınan gerçek zamanlı görüntülerdeki ASL (American Sign Language) el hareketlerini tanımayı amaçladım. Ayrıca gerçek zamanlı olarak algılanan el hareketlerinin cümle şeklinde yazılıp karşıdaki kişinin anlayabileceği yazı diline dönüştürmeyi hedefledim.

## 📸 Kullanılan Veri Seti
- Veri seti olarak **ASL Alphabet** veri setini kullandım.
- Veri seti [buradan](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) indirilebilir.

## 💻 Kullanılan Teknolojiler
- Python 3.11.8
- OpenCV: Kameradan gerçek zamanlı görüntüleri yakalamak için kullanıldı.
- PyTorch: Derin öğrenme modelini eğitmek için kullanıldı.
- Torchvision: Veri dönüşümlerini uygulamak için kullanıldı.
- MobileNetV2: Hafif ve hızlı çalışan CNN mimarisi. Gerçek zamanlı tahminler yaparken hız ve verimlilik sağladığı için tercih ettim.
- MediaPipe: Elin anahtar noktalarının çıkarılması için kullanılmıştır.
  
## ⚙️ Kurulum
GEREKLİ KÜTÜPHANELERİ KURUN
```bash
pip install tqdm
pip install torch torchvision torchaudio
pip install matplotlib
pip install scikit-image
pip install opencv-python
pip install numpy
pip install mediapipe
```

## 📸 Ekran Görüntüleri 
| 1 | 2 | 
| :---------------------------------: | :------------------------: |
|<img width="1061" height="844" alt="1" src="https://github.com/user-attachments/assets/99c3cf68-dc9b-423c-9d73-03f7ea426dfe" />| <img width="1058" height="848" alt="2" src="https://github.com/user-attachments/assets/2fad0908-0422-4731-bdb6-26665c292bfa" />


## 📺 Uygulama Videosu
▶️ [Watch Project Video on YouTube](https://www.youtube.com/watch?v=U1FSrVVDVnc)


## BU PROJE HİÇBİR ŞEKİLDE TİCARİ AMAÇ İÇERMEMEKTEDİR.

# [EN]

## Project Objective
In this project, I aimed to recognize ASL (American Sign Language) hand gestures in real-time from camera footage using the MobileNetV2 model. Additionally, I aimed to convert the detected hand gestures into written sentences in real-time, so that the other person can understand them as text.

## 📸 Dataset Used
- The dataset used is the ASL Alphabet dataset.
- The dataset can be downloaded [here](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

## 💻 Technologies Used
- Python 3.11.8
- OpenCV: Used to capture real-time images from the camera.
- PyTorch: Used to train the deep learning model.
- Torchvision: Used to apply data transformations.
- MobileNetV2: Lightweight and fast CNN architecture. Chosen for its speed and efficiency during real-time predictions.
- MediaPipe: Used to extract key points of the hand.

## ⚙️ Installation
INSTALL THE REQUIRED LIBRARIES
```bash
pip install tqdm
pip install torch torchvision torchaudio
pip install matplotlib
pip install scikit-image
pip install opencv-python
pip install numpy
pip install mediapipe
```
## 📸 Screenshots
| 1 | 2 | 
| :---------------------------------: | :------------------------: |
|<img width="1061" height="844" alt="1" src="https://github.com/user-attachments/assets/99c3cf68-dc9b-423c-9d73-03f7ea426dfe" />| <img width="1058" height="848" alt="2" src="https://github.com/user-attachments/assets/2fad0908-0422-4731-bdb6-26665c292bfa" />

## 📺 Demo Video
▶️ [Watch Project Video on YouTube](https://www.youtube.com/watch?v=U1FSrVVDVnc)

## THIS PROJECT IS IN NO WAY INTENDED FOR COMMERCIAL USE.
