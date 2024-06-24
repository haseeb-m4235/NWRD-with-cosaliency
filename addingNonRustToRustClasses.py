# Takes input a non rust directory and a rust directory
# The non rust diecotry contains all nonrust patches of a single image as a single group
# The rust directory contains groups named imgNo_partNo
# The code moves 2 non_rust patches of the same image to each rust group of that image

import glob
import os
from PIL import Image

rustDir = ""
nonRustDir = ""

rustClasses = os.listdir(rustDir+"/images/")
nonRustClassesNames = os.list(nonRustDir+"/images/")

nonRustCount = [0] * 150

for rustClass in rustClasses:
    for i in range(2):
        imgNo = rustClass.split("_")[0]
        nonRustPatchName = os.list(nonRustDir+ "/images/", f"/{imgNo}/")[nonRustCount[rustClass]+i]
        nonRustImagePath = os.path.join(nonRustDir, "images", imgNo, nonRustPatchName)
        image = Image.open(nonRustImagePath)
        imageNewPath = os.path.join(rustDir, "images", rustClass, f"nonRust{nonRustPatchName}.png")
        mask.save(imageNewPath)

        nonRustMaskPath = os.path.join(nonRustDir, "masks", imgNo, nonRustPatchName)
        mask = Image.open(nonRustMaskPath)
        maskNewPath = os.path.join(rustDir, "masks", rustClass, f"nonRust{nonRustPatchName}.png")
        mask.save(MaskNewPath)
        
        nonRustCount[rustClass] += 1
    

    
    

    