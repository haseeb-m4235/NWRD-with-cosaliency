import glob
import cv2
import numpy as np
from PIL import Image
import os

masks_paths = glob.glob('C:\\Users\\hasee\\Desktop\\NWRD  Internship\\FineLine\\codes\\patches 300\\testing\\masks_patches\\*')

for mask_path in masks_paths:
    #print("mask_path: ",mask_path)
    patches_masks_paths = glob.glob(mask_path+"\\*")
    image_name = mask_path.split("\\")[-1]
    #image_name = "rust"+image_name

    os.mkdir(f"C:\\Users\\hasee\\Desktop\\NWRD  Internship\\FineLine\\codes\\rust 300\\testing\\masks\\{image_name}")
    os.mkdir(f"C:\\Users\\hasee\\Desktop\\NWRD  Internship\\FineLine\\codes\\rust 300\\testing\\imgs\\{image_name}")

    dir_count=0
    patch_count=0

    for patch_mask_path in patches_masks_paths:
        patch_name =  patch_mask_path.split("\\")[-1].split(".")[0]

        # print("patch_name",patch_name)
        # print("image_name:", image_name)
        # print("patch_mask_path", patch_mask_path)
        # print("patch count:",patch_count)

        patch_mask = cv2.imread(patch_mask_path, 0)
        patch_img = cv2.imread(f"C:\\Users\\hasee\\Desktop\\NWRD  Internship\\FineLine\\codes\\patches 300\\testing\\images_patches\\{image_name}\\{patch_name}.JPG")
        #print("patch mask:", patch_mask[1])
        #print("patch image:", patch_img[1])

        if patch_count > 50:
            dir_count+=1
            new_image_name = image_name + str(dir_count)
            print("new image name:",new_image_name)
            os.mkdir(f"C:\\Users\\hasee\\Desktop\\NWRD  Internship\\FineLine\\codes\\rust 300\\testing\\masks\\{new_image_name}")
            os.mkdir(f"C:\\Users\\hasee\\Desktop\\NWRD  Internship\\FineLine\\codes\\rust 300\\testing\\imgs\\{new_image_name}")
            patch_count=0
            print("exceeds 50")

        condition = (patch_mask > 200)
        count = np.sum(condition)

        if count > 200:
            patch_count+=1
            print("patch_name",patch_name)
            print("image_name:", image_name)
            print("patch count:",patch_count)
            if (dir_count == 0):
                cv2.imwrite(f"C:\\Users\\hasee\\Desktop\\NWRD  Internship\\FineLine\\codes\\rust 300\\testing\\masks\\{image_name}\\{patch_name}.png", patch_mask)
                cv2.imwrite(f"C:\\Users\\hasee\\Desktop\\NWRD  Internship\\FineLine\\codes\\rust 300\\testing\\imgs\\{image_name}\\{patch_name}.png", patch_img)
            else: 
                cv2.imwrite(f"C:\\Users\\hasee\\Desktop\\NWRD  Internship\\FineLine\\codes\\rust 300\\testing\\masks\\{new_image_name}\\{patch_name}.png", patch_mask)
                cv2.imwrite(f"C:\\Users\\hasee\\Desktop\\NWRD  Internship\\FineLine\\codes\\rust 300\\testing\\imgs\\{new_image_name}\\{patch_name}.png", patch_img)
                
