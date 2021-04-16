import torch
import os
import glob
import pandas as pd
import torchvision.transforms as transforms
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader


class Melanoma_Dataset(Dataset):
    def __init__(self, groundtruth_path, image_path, transform = None, val = False):
        
        groundtruth_file_name = [z for _,_,z in os.walk(groundtruth_path)]
        self.groundtruth = pd.read_csv(groundtruth_path + "//" + groundtruth_file_name[0][0])
        self.image_path = image_path
        self.transform = transform
        
        self.composed = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
    def __len__(self):
        return len(self.groundtruth)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.image_path,
                               self.groundtruth.iloc[idx,0])
        image = io.imread(img_name)
        # labels = self.groundtruth.iloc[idx, 1:10]
        labels = torch.tensor(self.groundtruth.iloc[idx, 1:10].astype(int))
        # labels = np.array([labels])
        
        
        if self.transform:
            image = self.transform(image)
            
        
        return image, labels