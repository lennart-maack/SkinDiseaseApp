import os
import glob
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import transforms, utils
from skimage import io, transform
import numpy as np
import pandas as pd
from PIL import Image
import pretrainedmodels

import melDataset as melDataset
import nnModel as nnModel

class Predictor():
    
    def __init__(self):
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.255]
        self.input_path_predict_img = r"static/image_to_predict"
        self.groundtruth_path = r"static/predict_csv"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nnModel.SEResnext50_32x4d(pretrained=None)
        self.model_path = "model_8.bin"
        
    def __create_prediction(self, model, predict_loader, prediction_return=False):
        
        # Set to evaluation mode
        model.eval()
        with torch.no_grad():
            for i, (data, _) in enumerate(predict_loader):
                data= data.to(self.device)
                output = model(data)

                output = output.cpu().numpy()
                if i==0:
                    predictions = output
                    
                else:
                    predictions = np.concatenate((predictions,output))
        
        # predictions = np.argmax(predictions, 1)
        
        
        return predictions
    
    def __resize_image(self):
        # base_name = os.path.basename(self.input_path_predict_img)
        # outpath = os.path.join(output_folder, base_name)
        
        path = r"static/image_to_predict"
        
        image_names = [z for _,_,z in os.walk(self.input_path_predict_img)]
        
        img = Image.open(self.input_path_predict_img + "//" + image_names[0][0])
        img = img.resize(
            (512, 512), resample=Image.BILINEAR
        )
        img.save(path + "//" + image_names[0][0], "JPEG") 
    
    
    
    def __create_gt(self):
    
        image_names = [z for _,_,z in os.walk(self.input_path_predict_img)]

        dataframe = pd.DataFrame(columns=['MEL', "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"])

        dataframe.insert(loc=0, column="image", value=image_names[0])

        dataframe = dataframe.fillna(0)

        dataframe.to_csv("static/predict_csv/single_image_predict.csv", index=False)


    def predict(self):
        self.__resize_image()
        self.__create_gt()
        
        predict_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.255))])

        predict_dataset = melDataset.Melanoma_Dataset(groundtruth_path = self.groundtruth_path,
                                   image_path = self.input_path_predict_img,
                                   transform = predict_transform, val = False)
        
        predict_loader = torch.utils.data.DataLoader(dataset = predict_dataset, batch_size = 1,
                                                    shuffle = False)
        
        model_pred = self.model
        model_pred.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        model_pred.to(self.device)
        
        predictions = self.__create_prediction(model_pred, predict_loader)
        
        return predictions