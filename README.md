# Melanoma_Detection_CNN
Problem statement: 
To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution that can evaluate images and alert dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.

### The data set contains the following diseases:
### There are 9 classes of images
- Actinic keratosis
- Basal cell carcinoma
- Dermatofibroma
- Melanoma
- Nevus
- Pigmented benign keratosis
- Seborrheic keratosis
- Squamous cell carcinoma
- Vascular lesion

### Created train & validation dataset from the train directory
- data_dir_train = pathlib.Path("/Users/narayanaisanaka/UpGrad/2024/CNN/MEL/Skin cancer ISIC The International Skin Imaging Collaboration/Train")
- data_dir_test = pathlib.Path('/Users/narayanaisanaka/UpGrad/2024/CNN/MEL/Skin cancer ISIC The International Skin Imaging Collaboration/Test')

### Created a code to visualize one instance of all the nine classes present in the datase
- Total images for training: 2239
- Total images for test: 118

### Test Train split
Found 2239 files belonging to 9 classes.
Using 1792 files for training.
Using 447 files for validation.

### Explained the findings after the model fit with evidence if the model overfits or underfits.
Yes, Overfits were present in initial models.
- Train: 56/56 ━━━━━━━━━━━━━━━━━━━━ 2s 44ms/step - accuracy: 0.9017 - loss: 0.2568
- Test:  14/14 ━━━━━━━━━━━━━━━━━━━━ 1s 43ms/step - accuracy: 0.5547 - loss: 2.0418

### Chose an appropriate data augmentation strategy to resolve underfitting/overfitting
### With bit of Zoom this was resolved, but the accuracy of train came down.
- Train: 56/56 ━━━━━━━━━━━━━━━━━━━━ 6s 115ms/step - accuracy: 0.4530 - loss: 1.4801
- Test: 14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 114ms/step - accuracy: 0.4907 - loss: 1.4714

### Rectified class imbalances present in the training dataset with Augmentor library
- Used Augmentor lib to add more images to train/test dataset
### Test Train split
- Found 9041 files belonging to 9 classes.
- Using 7233 files for training.
- Using 1808 files for validation.

### Which class has the least number of samples?
- seborrheic keratosis


### Which classes dominate the data in terms of the proportionate number of samples?
- pigmented benign keratosis


### Explained the findings after the model fit with evidence if the issues are resolved or not.
-  A simple model was chosen with the similar accuracy with no overfit.
- Train: 227/227 ━━━━━━━━━━━━━━━━━━━━ 15s 65ms/step - accuracy: 0.4706 - loss: 1.4401
- Test:  57/57 ━━━━━━━━━━━━━━━━━━━━ 4s 64ms/step - accuracy: 0.4742 - loss: 1.4332

### Ovefit Model on less data
227/227 ━━━━━━━━━━━━━━━━━━━━ 12s 52ms/step - accuracy: 0.9584 - loss: 0.1213
57/57 ━━━━━━━━━━━━━━━━━━━━ 3s 54ms/step - accuracy: 0.8130 - loss: 0.9264

### Retried the overfit model again with large data
- Train: 227/227 ━━━━━━━━━━━━━━━━━━━━ 15s 65ms/step - accuracy: 0.9314 - loss: 0.1902
- Test: 57/57 ━━━━━━━━━━━━━━━━━━━━ 4s 64ms/step - accuracy: 0.7699 - loss: 1.0851



### Packages used:
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import PIL
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import warnings
warnings.filterwarnings('ignore')
Augmentor
