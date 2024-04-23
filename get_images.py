'''
This function, `get_images`, is designed to load images from a 
specified directory and its subfolders, resize them using OpenCV, and 
then return the resized images along with their corresponding labels. 
It iterates through each subfolder in the given directory, checks for 
valid image files (with extensions jpg, png, or gif), reads each image 
using OpenCV, resizes them to a specified size (100x100 pixels in this 
case), and stores both the resized images and their labels in separate 
lists (`X` for images and `y` for labels). Finally, it prints a message 
indicating that all images have been loaded and returns the lists of 
images and labels.
'''


import os
import cv2

def get_images(image_directory):
    # Initialize lists to store images and their labels
    X = []
    y = []
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
                    # Read the image using OpenCV
                    img = cv2.imread(os.path.join(image_directory, subfolder, file))
        
                    # Resize the image
                    img = cv2.resize(img, (100, 100))
                    
                    # Add the resized image to the list X
                    X.append(img)
                    
                    # Add the image's label to the list y
                    y.append(subfolder)
    
    print("All images are loaded")
    return X, y


X, Y = get_images("C:/Users/Ethan/Documents/GitHub/Face-and-Keystroke-Authentication/project_data/project_data")
