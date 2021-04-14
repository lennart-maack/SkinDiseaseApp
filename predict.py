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


mean = [0.485, 0.456, 0.406]

std = [0.229, 0.224, 0.255]

input_path_predict_img = r"static\image_to_predict"

groundtruth_path = r"static\predict_csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = nnModel.SEResnext50_32x4d(pretrained=None)

model_path = "model_8.bin"

    


def predict(model, device, predict_loader, prediction_return=False):
    # Set to evaluation mode
    model.eval()
    with torch.no_grad():
        for i, (data, _) in enumerate(predict_loader):
            data= data.to(device)
            output = model(data)

            output = output.cpu().numpy()
            if i==0:
                predictions = output
                
            else:
                predictions = np.concatenate((predictions,output))
    
    # predictions = np.argmax(predictions, 1)
    
    
    return predictions


def predict_SEResnext50():
    
    model_pred = model
    model_pred.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model_pred.to(device)
    
    new_predictions = predict(model_pred, device, predict_loader)
    
    return new_predictions


def create_gt():
    
    image_names = [z for _,_,z in os.walk(input_path_predict_img)]

    dataframe = pd.DataFrame(columns=['MEL', "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"])

    dataframe.insert(loc=0, column="image", value=image_names[0])

    dataframe = dataframe.fillna(0)

    dataframe.to_csv("static/predict_csv/single_image_predict.csv", index=False)
    

def resize_image(image_path, output_folder):
    base_name = os.path.basename(image_path)
    outpath = os.path.join(output_folder, base_name)
    
    path = r"static\image_to_predict"
    
    image_names = [z for _,_,z in os.walk(input_path_predict_img)]
    
    img = Image.open(image_path + "\\" + image_names[0][0])
    img = img.resize(
        (512, 512), resample=Image.BILINEAR
    )
    img.save(path + "\\" + image_names[0][0], "JPEG")    


resize_image(input_path_predict_img, input_path_predict_img)

create_gt()


predict_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.255))])


predict_dataset = melDataset.Melanoma_Dataset(groundtruth_path = groundtruth_path,
                                   image_path = input_path_predict_img,
                                   transform = predict_transform, val = False)

predict_loader = torch.utils.data.DataLoader(dataset = predict_dataset, batch_size = 1,
                                             shuffle = False)




new_predictions = predict_SEResnext50()

print(new_predictions)