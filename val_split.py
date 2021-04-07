import pandas as pd
from joblib import Parallel,delayed
import shutil, os
import glob
from tqdm import tqdm

    """
    this val_split.py script is used to split the 2019 Training Data
    from https://challenge.isic-archive.com/data according to the groundtruth files into train and val
    """



# Set your local path the the two ground truth csv files that can be downloaded from the studIP group
gt_train_path = r""
gt_val_path = r""

# Set the local full path where you store train images
old_train_image_path = r""

# Set the local full path where you want to store the newly split up train images and val images (according to the gt data)
new_train_image_path = r""
new_val_image_path = r""

# Read in the ground truth data to pandas df
train_groundtr = pd.read_csv(gt_train_path)
val_groundtr = pd.read_csv(gt_val_path)

# Make a list out if the image column names from the created df
gt_train_image_ids = train_groundtr.image.values.tolist()
gt_val_image_ids = val_groundtr.image.values.tolist()

# Make a list of all the image names from the folder where you stored ALL train images
jpg_image_ids = glob.glob1(old_train_image_path, "*.jpg")

# Check if the lists have the correct lengths
print(len(jpg_image_ids))
print(len(gt_train_image_ids))
print(len(gt_val_image_ids))


# Define function to move the new train images into a new train folder
# You could also use joblib to speed things up and parallelize
def move_traindata_into_new_folder():
    
    for filename in tqdm(glob.glob1(old_train_image_path, "*.jpg"), total=len(jpg_image_ids)):
        if filename[:-4] in gt_train_image_ids:
            count += 1
            shutil.copy(os.path.join(old_train_image_path, filename), new_train_image_path)



# Define function to move the new val images into a new val folder
def move_valdata_into_new_folder():
    for filename in tqdm(glob.glob1(old_train_image_path, "*.jpg"), total=len(jpg_image_ids)):
        if filename[:-4] in gt_val_image_ids:
            count += 1
            shutil.copy(os.path.join(old_train_image_path, filename), new_val_image_path)

