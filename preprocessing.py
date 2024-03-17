import cv2
from PIL import Image
import numpy as np
from mtcnn import MTCNN
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# import sys
# sys.path.append("MTCNN")

#Define url
root_url = os.path.join(os.path.abspath(os.path.dirname(__name__)),"MTCNN") #Change root_url according to computer directory
root_raw_img = os.path.join(root_url,"Raw_images")
detector = MTCNN()


def display_image(imgfile):
    """
    Display detected face after using MTCNN
    """
    ax = plt.gca()
    plt.imshow(imgfile)
    img_array = np.array(imgfile)
    plt.axis("off")
    results = detector.detect_faces(img_array)
    for result in tqdm(results): 
        print(result)
        if result['confidence'] > 0.9:
            x,y,width,height = result['box'] 
            rect = Rectangle((x,y),width,height,fill = False, color = 'green')
            ax.add_patch(rect)

    for _,value in result['keypoints'].items():
        circle = Circle(value, radius = 2, color = 'red')
        ax.add_patch(circle)
    plt.show()


def face_extract(file_name, target_size = (160, 160)):
    """
    Face extraction from an image directory
    Input: Image directory as string
    Output: Resized image, array of resize image
    """
    img = Image.open(file_name)
    img_arr = np.asarray(img)
    result = detector.detect_faces(img_arr)
    if len(result) == 0:
        return None, None
    else:
        x1, y1, width, height = result[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = img_arr[y1:y2, x1:x2]
        image = Image.fromarray(face)
        image = image.resize(target_size)
        resized_arr = np.asarray(image)
    return image, resized_arr


def split_resized(root_url, resized_face, resized_url, name, test_size):
    #Resized_face is a list of img filename(Ex: a.png)
    """
    Split resized images into train/test folder for one class
    Input: 
    - root_url: Global varible defined earlier
    - resized_face:
    """
    # Create train/test folders
    os.makedirs(os.path.join(root_url, "Train"), exist_ok=True)
    os.makedirs(os.path.join(root_url, "Test"), exist_ok=True)

    train, test = train_test_split(resized_face, test_size = test_size, random_state= 42, shuffle= True)
    # Create the folder if it doesn't exist
    folder_path = os.path.join(root_url, "Train", name)
    os.makedirs(folder_path, exist_ok=True)
    #Save train img
    # Iterate through each image in the train list
    for train_img in tqdm(train):
        # Get the path of the original image and destination folder
        src_path = os.path.join(resized_url, name, train_img)
        dest_path = os.path.join(root_url, "Train", name, train_img)
        
        # Open the image
        img = Image.open(src_path)
        
        # Save the image to the destination folder
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        img.save(dest_path, format="PNG")

    folder_path = os.path.join(root_url, "Test", name)
    os.makedirs(folder_path, exist_ok=True)
    #Save test img
    for test_img in tqdm(test):
        # Get the path of the original image and destination folder
        src_path = os.path.join(resized_url, name, test_img)
        dest_path = os.path.join(root_url, "Test", name, test_img)
        
        # Open the image
        img = Image.open(src_path)
        
        # Save the image to the destination folder
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        img.save(dest_path, format="PNG")

# def train_test(resized_face):
#     train_set, test_set = train_test_split(resized_face, test_size = 0.3)
#     return train_set, test_set


#Function to extract faces, resize and save to resized folder of dataset
def extract_face_fromdir(name_list):
    """
    Extract faces, resize and save to resized folder of dataset (for further uses)
    Input: 
    - name_list: List of classes
    Output:
    - file_name_list: Store img filename (directory)
    - file_dict: Dictionary which keys are class and values are np.array of resized image respectively.
    """
    face_dict = {}       #Store img array
    file_name_list = {}  #Store img filename
    # Create dictionary to store resized img
    for name in tqdm(name_list):
        face_dict[name] = []
        file_name_list[name] = []
    for name in tqdm(name_list):
        path = os.path.join(root_raw_img, name)
        for _, img in tqdm(enumerate(os.listdir(path))):
            img_path = os.path.join(path, img)
            file_name = os.path.basename(img_path)
            file_name_list[name].append(file_name)
            file_img, resized_img = face_extract(img_path)
            if (file_img == None and resized_img == None):
                continue
            face_dict[name].append(resized_img)
            resized_dir = os.path.join(root_url, "Rezised", name)
            os.makedirs(resized_dir, exist_ok=True)
            file_img.save(os.path.join(resized_dir, img), format="png")
    return file_name_list, face_dict  


def train_test_seperate(name_list):
    """
    Split 
    """
    path = os.path.join(root_url,"Rezised")
    for name in tqdm(name_list):
        imgs_path = os.path.join(path, name)
        resized_faces = [f for f in os.listdir(imgs_path) if os.path.isfile(os.path.join(imgs_path, f))]
        split_resized(root_url, resized_faces, path, name, test_size=0.3)


def normalize(face_picels):
    """
    Normalize image's pixel value
    """
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    return face_picels

# if __name__ == "__main__":
#     extract_face_fromdir(name_list)
#     train_test_seperate()
