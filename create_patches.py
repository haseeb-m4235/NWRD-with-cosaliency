from functools import total_ordering
import glob
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image


def create_patches(fname):
    x = 0
    y = 0
    patches = []
    img = cv2.imread(fname)
    print("image shape:",img.shape)
    p_num = 0
    while (y + patch_size < img.shape[0]):
        
        if (x + patch_size > img.shape[1]):
            x = 0
            y += patch_size
        if y + patch_size <= img.shape[0] and x + patch_size <= img.shape[1]:
            patches.append([x, y])
        x += patch_size
    print("total patches: ", len(patches))
    return patches

# def create_patches(fname):
#     # Read the image
#     img = cv2.imread(fname)
#     height, width = img.shape[:2]
#     print("Original image shape:", img.shape)

#     # Calculate the padding needed
#     right_padding = (patch_size - width % patch_size) % patch_size
#     bottom_padding = (patch_size - height % patch_size) % patch_size

#     # Pad the image with black pixels
#     img_padded = cv2.copyMakeBorder(img, 0, bottom_padding, 0, right_padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])
#     print("Padded image shape:", img_padded.shape)

#     # Create patches
#     patches = []
#     y = 0
#     while (y + patch_size <= img_padded.shape[0]):
#         x = 0
#         while (x + patch_size <= img_padded.shape[1]):
#             # Extract the patch
#             patch = img_padded[y:y+patch_size, x:x+patch_size]
#             patches.append(patch)
#             x += patch_size
#         y += patch_size

#     print("Total patches:", len(patches))
#     return patches


patch_size = 300
images_paths = glob.glob('C:\\Users\\hasee\\Desktop\\NWRD  Internship\\FineLine\\NWRD\\test\\images\\*')
masks_paths = glob.glob('C:\\Users\\hasee\\Desktop\\NWRD  Internship\\FineLine\\NWRD\\test\\masks\\*')

images_paths.sort()
masks_paths.sort()
print("frist image path: ", images_paths[0])
print("frist mask path: ", masks_paths[0])


total_count = 0
for u in images_paths:
    print(u)
    patches = create_patches(u)
    bgr = cv2.imread(u)
    image_name = u.split('\\')[-1].split('.')[0]
    os.makedirs(f"images_patches/{image_name}", exist_ok=True)
    total_count += len(patches)

    for count, P in enumerate(patches):
        cv2.imwrite(f"images_patches/{image_name}/{count}.JPG", bgr[P[1]:P[1]+patch_size,P[0]:P[0]+patch_size])
            
print("total count:", total_count)

total_count = 0
for u in masks_paths:
    print(u)
    patches = create_patches(u)
    bgr = cv2.imread(u)
    image_name = u.split('\\')[-1].split('.')[0]
    os.makedirs(f"masks_patches/{image_name}", exist_ok=True)
    total_count += len(patches)

    for count, P in enumerate(patches):
        cv2.imwrite(f"masks_patches/{image_name}/{count}.JPG", bgr[P[1]:P[1]+patch_size,P[0]:P[0]+patch_size])
            
print("total masks count:", total_count)
