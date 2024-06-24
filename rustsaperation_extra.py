import glob
import cv2
import numpy as np
from PIL import Image
import os

root = "C:\\Users\\hasee\\Desktop\\Germany_2024\\Dataset\\NWRDPatches\\train"
destination = "C:\\Users\\hasee\\Desktop\\Germany_2024\\Dataset\\NWRDClassifier\\train"
masks_path = os.path.join(root, "masks\\*.png")
masks_paths = glob.glob(masks_path)
minimum=1000
min_patch=0
rust_count=0
non_rust_count=0

for mask_path in masks_paths:
    print("mask_path: ",mask_path)
    patch_name = mask_path.split("\\")[-1].split(".")[0]
    
    patch_mask = cv2.imread(mask_path, 0)
    patch_img = cv2.imread(os.path.join(root, "images",patch_name+".png"))

    condition = (patch_mask > 150)
    count = np.sum(condition)
        
    if count<=150:
            #os.remove(os.path.join(root, "images",patch_name+".png"))
        #     os.remove(mask_path)
            #cv2.imwrite(os.path.join(destination,f"non_rust/{patch_name}.png"), patch_img)
           # cv2.imwrite(os.path.join(destination,f"non_rust/{patch_name}.png"), patch_img)
            cv2.imwrite(os.path.join(destination,f"non_rust\\images\\{patch_name}.png"), patch_img)
            cv2.imwrite(os.path.join(destination,f"non_rust\\masks\\{patch_name}.png"), patch_mask)
            non_rust_count+=1
    else:
            if (count<=minimum):
                   minimum=count
                   min_patch = patch_name
            #cv2.imwrite(os.path.join(destination,f"rust/{patch_name}.png"), patch_img)
            #cv2.imwrite(os.path.join(destination,f"rust/{patch_name}.png"), patch_img)
            cv2.imwrite(os.path.join(destination,f"rust\\images\\{patch_name}.png"), patch_img)
            cv2.imwrite(os.path.join(destination,f"rust\\masks\\{patch_name}.png"), patch_mask)
            rust_count+=1

print("minimum rust patch:",min_patch)
print("minimum rust patch white pixels:",minimum)
print("rust count=", rust_count)
print("non rust count=", non_rust_count)



                
