"""FedRGBD — PyTorch Dataset for FLAME fire classification."""

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FlameDataset(Dataset):
    """FLAME Fire/NoFire classification dataset."""
    
    def __init__(self, data_dir, split="train", img_size=224):
        self.data_dir = os.path.join(data_dir, split)
        self.img_size = img_size
        self.samples = []
        self.class_to_idx = {"No_Fire": 0, "Fire": 1}
        
        for class_name, label in self.class_to_idx.items():
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(class_dir, fname), label))
        
        if split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label
    
    def get_class_distribution(self):
        counts = {name: 0 for name in self.class_to_idx}
        for _, label in self.samples:
            for name, idx in self.class_to_idx.items():
                if idx == label:
                    counts[name] += 1
        return counts


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/processed/iid/node_a"
    for split in ["train", "val", "test"]:
        ds = FlameDataset(data_dir, split=split)
        dist = ds.get_class_distribution()
        print(f"{split}: {len(ds)} samples — {dist}")
