from functools import total_ordering
import glob
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
import math



patch_size = (300, 300)
def get_patch_position(img_shape, patch_size, patch_no):
    num_patches_horizontally = img_shape[1] // patch_size[1]
    num_patches_vertically = img_shape[0] // patch_size[0]
    
    # Check if patch_no is within the valid range
    if patch_no >= num_patches_horizontally * num_patches_vertically:
        raise ValueError("Invalid patch number for the given image size and patch size.")
    
    # Calculate row and column starting from 0
    row = patch_no // num_patches_horizontally
    col = patch_no % num_patches_horizontally
    
    return row, col

def replace_with_patch(matrix, patch, row, col, patch_size):
    start_y = row * patch_size[0]
    start_x = col * patch_size[1]

    end_y = start_y + patch_size[0]
    end_x = start_x + patch_size[1]

    # Adjust end positions if they exceed matrix dimensions
    end_y = min(end_y, matrix.shape[0])
    end_x = min(end_x, matrix.shape[1])

    # Adjust patch size if necessary
    patch = patch[:end_y - start_y, :end_x - start_x]

    matrix[start_y:end_y, start_x:end_x] = patch
    return matrix

# def get_patch_position(img_shape, patch_size, patch_no):
#     # Calculate the number of patches that fit horizontally and vertically
#     num_patches_horizontally = img_shape[1] // patch_size[1]
#     num_patches_vertically = img_shape[0] // patch_size[0]
    
#     # Ensure the given patch number is valid
#     if patch_no > num_patches_horizontally * num_patches_vertically:
#         raise ValueError("Invalid patch number for the given image size and patch size.")
    
#     # Calculate row and column based on the patch number
#     row = (patch_no - 1) // num_patches_horizontally + 1
#     col = (patch_no - 1) % num_patches_horizontally + 1
    
#     return row, col

# def replace_with_patch(matrix, patch, row, col, patch_size):
#     start_y = (row - 1) * patch_size[0]
#     start_x = (col - 1) * patch_size[1]
#     matrix[start_y:start_y+patch_size[0], start_x:start_x+patch_size[1]] = patch
#     return matrix


patch_size = (300, 300)
masks = glob.glob('C:\\Users\\hasee\Desktop\\NWRD  Internship\\FineLine\\FInal results\\vgg_cosal_results\\*')
#print(masks)
for mask in masks:
    patches = glob.glob(f"{mask}\\*")
    #print(patches)
    imgNo = mask.split('\\')[-1]
    #print(imgNo)"C:\Users\hasee\Desktop\NWRD  Internship\FineLine\NWRD\test\masks\25.png"
    orig_mask_path = f"C:\\Users\\hasee\\Desktop\\NWRD  Internship\\FineLine\\NWRD\\test\\masks\\{imgNo}.png"
    print("Original mask path:", orig_mask_path)
    img = cv2.imread(orig_mask_path, 0)
    print(img)
    print("image shape:", img.shape)
    pred = np.zeros(img.shape)
    for patch_path in patches:
        #patchNo = patch_path.split('\\')[-1][:-4]
        patchNo = patch_path.split('\\')[-1][:-4].split('.')[0]
        print("pathcNo:",patchNo)
        row, col = get_patch_position(img.shape, patch_size, int(patchNo))

        patch = cv2.imread(patch_path, 0)
        print("patch original shape", patch.shape) 
        patch = cv2.resize(patch, (300, 300), interpolation=cv2.INTER_AREA)
        print("resized patch shape", patch.shape)
         
        print(f"Row: {row}, Column: {col}")  
        pred = replace_with_patch(pred, patch, row, col, patch_size)
    save_dir = f"C:\\Users\\hasee\\Desktop\\NWRD  Internship\\FineLine\\FInal results\\modified_concatenation_resized_results\\{imgNo}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cv2.imwrite(os.path.join(save_dir, f"{imgNo}.png"), pred)


