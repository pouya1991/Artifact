#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yuri tolkach
"""

# =============================================================================
# 1. Focus artefacts (Gaussian blur)
# =============================================================================

#Parameters
#Number of gaussian levels to test
num_g_lev = 16
#Directory for files (dataset)
source_dir = '../images/'
#Output directory for results
output_dir = './output/dir/'

path_result = output_dir + "01_focus.txt"
#Model directory (pre-tranied prostate cancer detection model)
model_dir = '../model/'
model_name = 'model_2.pt'


#Load necessary libraries
import cv2
import os
import numpy as np
# from tensorflow.keras.models import load_model
import torch
from torchvision import transforms

#Load model
path_model = os.path.join(model_dir, model_name)
model = torch.load(path_model, map_location=torch.device('cpu'))
model.model.eval()
print(model.model)
os.makedirs(output_dir, exist_ok=True)



transform = transforms.Compose([transforms.ToTensor()])
#Function for classification prediction using a model
def predict (patch):
    img_t = transform(patch/255).float()
    # wp_temp = np.float32(patch)
    img_t = torch.unsqueeze(img_t, 0)
    # wp_temp /= 255.
    with torch.no_grad():
        preds = model.model.forward(img_t)
    return preds.cpu().detach().numpy().tolist()

#Function to write result into output txt file
def write_result (output, path_result):
    results = open (path_result, "a+")
    results.write(output)
    results.close()       

#Read subdirectory names
dir_names = sorted(os.listdir(source_dir)) # gland, nongland, tumor

for dir_name in dir_names:
    work_dir = source_dir + dir_name + "/"    
    filenames = os.listdir(work_dir)
        
    #Loop for opening of single files, generating artifact, and
    #making classification predictions.
    for filename in filenames:
        image = cv2.imread(work_dir+filename)
        image = cv2.resize(image, (300, 300), cv2.INTER_AREA)
        print('loaded', filename)
        preds_all = ''
        preds_all = filename + "\t"
        for i in range(1, 2 * num_g_lev, 2):
            image_blur = cv2.GaussianBlur(image, (i, i), 0)
            image_blur = cv2.cvtColor(image_blur, cv2.COLOR_BGR2RGB)
            preds = predict(image_blur)
            print(preds)
            print("\t".join([str(round(pred, 3)) for pred in preds[0]]))
            preds_all = preds_all + "\t".join([str(round(pred, 3)) for pred in preds[0]]) + "\t"
        preds_all = preds_all + "\n"
        write_result(preds_all, path_result)
            


