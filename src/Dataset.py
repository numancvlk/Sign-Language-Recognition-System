#LIBRARIES
import os
from torch.utils.data import Dataset
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, rootDir, tranform = None, labeled = True):
        """
        root_dir: klasör yolu (örnek: data/train veya data/test)
        transform: torchvision.transforms nesnesi
        labeled: True -> alt klasörler etiket (train)
                 False -> direkt resimler (test)
        """

        self.rootDir = rootDir
        self.transform = tranform
        self.labeled = labeled
        self.samples = []

        if labeled:
            # Alt klasörleri gez (örnek: A/, B/, C/...)
            classes = sorted(os.listdir(rootDir))
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

            for cls_name in classes:
                cls_folder = os.path.join(rootDir, cls_name)
                if not os.path.isdir(cls_folder):
                    continue
                for file_name in os.listdir(cls_folder):
                    if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append({
                            "path": os.path.join(cls_folder, file_name),
                            "label": self.class_to_idx[cls_name]
                        })
        else:
            # Etiketsiz test dosyaları
            self.samples = [
                {"path": os.path.join(rootDir, f), "label": None}
                for f in os.listdir(rootDir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["path"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.labeled:
            return image, sample["label"]
        else:
            # test verisinde etiket yok -> sadece image ve dosya adı
            return image, os.path.basename(sample["path"])