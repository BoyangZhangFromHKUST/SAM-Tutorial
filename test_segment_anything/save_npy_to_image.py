import numpy as np
import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt

folder_path = "./for_gaorongrong_farbic"

# Get a list of all .npy files in the folder
file_list = [f for f in os.listdir(folder_path) if f.endswith("_c.npy")]

for file_name in file_list:
    
    #read depth
    depth_path = os.path.join(folder_path, file_name.split('_')[0]+"_d.npy")
    depth_array = np.load(depth_path)
    # depth_array[np.where(depth_array>0.475)] =1
    # plt.imshow(depth_array)
    
    # Read the rgb .npy file
    file_path = os.path.join(folder_path, file_name)
    image_array = np.load(file_path)
    # image_array[np.where(depth_array>0.48)] = 255
    plt.imshow(image_array)
    
    # Convert BGR to RGB
    image_array_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    
    # Convert the numpy array to PIL Image
    image = Image.fromarray(image_array_rgb)
    
    # Save the image as .png with "_c" suffix
    save_path = os.path.splitext(file_path)[0] + ".png"
    image.save(save_path)
    print(f"Saved {save_path}")
print("Images saved successfully!")
