import torch
from sklearn.metrics import f1_score as f1
from sklearn.metrics import precision_score, recall_score
import os
from PIL import Image
import numpy as np

def precision(predicted_mask, ground_truth_mask):
    with torch.no_grad():
        assert predicted_mask.shape == ground_truth_mask.shape
        return precision_score(ground_truth_mask.view(-1).cpu().numpy(), predicted_mask.view(-1).cpu().numpy())

def recall(predicted_mask, ground_truth_mask):
    with torch.no_grad():
        assert predicted_mask.shape == ground_truth_mask.shape
        return recall_score(ground_truth_mask.view(-1).cpu().numpy(), predicted_mask.view(-1).cpu().numpy())

def f1_score(predicted_mask, ground_truth_mask):
    with torch.no_grad():
        assert predicted_mask.shape == ground_truth_mask.shape
        return f1(ground_truth_mask.view(-1).cpu().numpy(), predicted_mask.view(-1).cpu().numpy())

def load_images_from_directory(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            print(f'Filename:{filename}')
            image_path = os.path.join(directory, filename)
            image = Image.open(image_path)
            image = np.array(image)
            threshold = 128
            image = (image > threshold).astype(np.float32)
            image = torch.tensor(image, dtype=torch.float32)
            images.append(image)
    return images

# Load predicted and ground truth masks from directories
predicted_masks_directory = "C:\\Users\\hasee\\Desktop\\NWRD  Internship\\FineLine\\FInal results\\modified_concatenation_resized_results"
ground_truth_masks_directory = "C:\\Users\\hasee\\Desktop\\NWRD  Internship\\FineLine\\NWRD\\test\\masks"

predicted_masks = load_images_from_directory(predicted_masks_directory)
ground_truth_masks = load_images_from_directory(ground_truth_masks_directory)

precisions = []
recalls = []
f1_scores = []

for predicted_mask, ground_truth_mask in zip(predicted_masks, ground_truth_masks):
    print(f'predicted_mask:{np.shape(predicted_mask)} ground_truth_mask:{np.shape(ground_truth_mask)}')
    precision_value = precision(predicted_mask, ground_truth_mask)
    recall_value = recall(predicted_mask, ground_truth_mask)
    f1_score_value = f1_score(predicted_mask, ground_truth_mask)
    
    precisions.append(precision_value)
    recalls.append(recall_value)
    f1_scores.append(f1_score_value)

# Calculate average metrics
average_precision = sum(precisions) / len(precisions)
average_recall = sum(recalls) / len(recalls)
average_f1_score = sum(f1_scores) / len(f1_scores)

print("Average Precision:", average_precision)
print("Average Recall:", average_recall)
print("Average F1 Score:", average_f1_score)
