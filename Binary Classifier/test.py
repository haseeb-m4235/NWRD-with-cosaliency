import torch
from torchvision import transforms
import torch
import glob
import cv2
from torch import nn
from torchvision import models
import numpy as np
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
from skimage import io
import os   
def get_dataset(masks_dir):
    rust = []
    non_rust = []
    masks_paths = glob.glob(masks_dir+'*')
    for mask_path in masks_paths:
        mask = cv2.imread(mask_path, 0) 
        condition = (mask > 200)
        count = np.sum(condition)
        if count > 200:
            rust.append(mask_path.replace("masks", "images"))
        else:
            non_rust.append(mask_path.replace("masks", "images"))
    return rust, non_rust

class CData(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        img_path = self.x[index]
        image = io.imread(img_path)
        label = int(self.y[index])
        if self.transform:
            image = self.transform(image)
        return (image, label, img_path)
    
transform_test = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor()])


model = models.vgg16(pretrained=True)
model.classifier[6] = nn.Linear(4096, 2)
model = model.cpu()

custom_state_dict_path = "/home2/haseeb_muhammad/BinaryClassifier/vgg16_300_patch_3rdtry/model_49.pth"  # Provide the path to your custom state dictionary
custom_state_dict = torch.load(custom_state_dict_path, map_location=torch.device('cpu'))
model.load_state_dict(custom_state_dict)

rust, non_rust = get_dataset("/home2/haseeb_muhammad/BinaryClassifier/Combined_images/test/masks/")
rust_y = [1]*len(rust)
non_rust_y = [0]*len(non_rust)

x_test = rust
x_test.extend(non_rust)
y_test = rust_y
y_test.extend(non_rust_y)

x_test, y_test = np.array(x_test), np.array(y_test)

print(f"x_test: {len(x_test)} y_test: {len(y_test)}")

test_dataset = CData(x_test, y_test, transform=transform_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

# Set the model to evaluation mode
model.eval()

# Variables to track correct predictions and total predictions
correct = 0
total = 0
tp = 0
fp = 0
fn = 0
tn = 0
# No need to track gradients for evaluation
with torch.no_grad():
    for data in test_loader:
        images, labels, img_paths = data
        print(img_paths)
        image_name = img_paths[0].split('/')[-1].split('_')[0]
        patch_name = img_paths[0].split('/')[-1].split('_')[-1]
        images = Variable(images).cpu()
        labels = Variable(labels).cpu()

        # Forward pass to get outputs
        outputs = model(images)
        print(outputs.data)

        # Get predictions from the maximum value
        _, predicted = torch.max(outputs.data, 1)
        if predicted == 1:
            print("image_name:",image_name)
            print("patch_name:",patch_name)
            output_path = f"/home2/haseeb_muhammad/BinaryClassifier/Result/{image_name}/{patch_name}.png"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            image_array = images[0].cpu().numpy()
            image_array = np.transpose(image_array, (1, 2, 0))
            image_array = (image_array * 255).astype(np.uint8)
            cv2.imwrite(output_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))

        # Total number of labels
        total += labels.size(0)

        if predicted == 0 and labels == 0:
            tn +=1
        elif predicted == 0 and labels == 1:
            fn +=1
        if predicted == 1 and labels == 1:
            tp +=1
        if predicted == 1 and labels == 0:
            fp +=1

        # Total correct predictions
        correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print("Accuracy: ", accuracy)
        print("Total: ", total)
        print(f"True Positives: {tp}  False Positives: {fp}  True Negative: {tn}  False Negative: {fn}")

# Calculate the accuracy
accuracy = 100 * correct / total
print("Final Resutls: ")
print(f'Accuracy of the model on the test images: {accuracy}%')
print(f"True Positives: {tp}  False Positives: {fp}  True Negative: {tn}  False Negative: {fn}")