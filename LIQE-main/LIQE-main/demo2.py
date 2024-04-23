import torch
import numpy as np
import random
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from LIQE import LIQE
from torchvision.transforms import ToTensor
from pathlib import Path
seed = 20200626
num_patch = 15
import os

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
ckpt = './checkpoints/LIQE.pt'
model = LIQE(ckpt, device)

img1 = 'C:/Users/Ethan/Documents/GitHub/Face-and-Keystroke-Authentication/LIQE-main/LIQE-main/data/Arleo-Gil_1_frame_0_FACE1.jpg'

#imgPath = Path("C:\Users\Ethan\Documents\GitHub\Face-and-Keystroke-Authentication\project_data\project_data\Arleo-Gil\Arleo-Gil_1_frame_0_FACE1.jpg")
print('###Image loading###')

I1 = Image.open(img1)

newsize = (300,300)
I1 = I1.resize(newsize)

I1 = ToTensor()(I1).unsqueeze(0)


print('###Preprocessing###')
with torch.no_grad():
    q1, s1, d1 = model(I1)

print(f"q1 = {q1}, s1 = {s1}, d1 = {d1}")
print(q1.item())

print('Image #1 is a photo of {} with {} artifacts, which has a perceptual quality of {} as quantified by LIQE'.
      format(s1, d1, q1.item()))


def assess_image_quality(path):
    seed = 20200626
    num_patch = 15

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    ckpt = './checkpoints/LIQE.pt'
    model = LIQE(ckpt, device)

    image_directory = path
    # Initialize lists to store images and their labels
    X = []
    y = []
    qualityScores = []
    # Define image file extensions
    extensions = ('jpg', 'png', 'gif')
    
    # Iterate through subfolders in the given directory
    for subfolder in os.listdir(image_directory):
        print("Loading images in %s" % subfolder)
        
        # Check if the item is a directory
        if os.path.isdir(os.path.join(image_directory, subfolder)):
            # Get the list of files in the subfolder
            subfolder_files = os.listdir(os.path.join(image_directory, subfolder))
            
            # Iterate through files in the subfolder
            for file in subfolder_files:
                # Check if the file has a valid image extension
                if file.endswith(extensions):
                    # Read the image
                    I1 = Image.open(os.path.join(image_directory, subfolder, file))
                    print("new image opened")

                    newsize = (300,300)
                    I1 = I1.resize(newsize)

                    I1 = ToTensor()(I1).unsqueeze(0)
                    with torch.no_grad():
                        q1, s1, d1 = model(I1)
                    qualityScores.append(q1.item())
                    
                  
                    
                    # Add the image's label to the list y
                    y.append(subfolder)
    return qualityScores, y

qualityScores, y = assess_image_quality("C:/Users/Ethan/Documents/GitHub/Face-and-Keystroke-Authentication/project_data/project_data")
print(qualityScores)
np.save('data.npy', qualityScores)