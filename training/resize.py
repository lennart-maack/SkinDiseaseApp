import os
import glob
from tqdm import tqdm
from PIL import Image
from joblib import Parallel, delayed

#Parameter
height = 32
width = 32

# Set path to input_folder and output_folder
input_folder = r""
output_folder = r""

images = glob.glob(os.path.join(input_folder, "*.jpg"))


def resize_image(image_path, output_folder, resize):
    base_name = os.path.basename(image_path)
    outpath = os.path.join(output_folder, base_name)
    img = Image.open(image_path)
    img = img.resize(
        (resize[1], resize[0]), resample=Image.BILINEAR
    )
    img.save(outpath)


Parallel(n_jobs=12)(
    delayed(resize_image)(
        i, output_folder, (height, width)
    ) for i in tqdm(images)
)
