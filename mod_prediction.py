import os
from tensorflow import zeros

from contextlib import redirect_stdout

import models
import utils
from settings import IG_FOLDER, cmap
#from ig import integrated_gradients
from preproc import read_image
from models import get_pred_class_idx_k

print('-----SETUP-----')
# Setup
utils.setup_folder_structure()

print('-----LOAD MODELS-----')
# Load models
##inception_v1
#model = models.load_inception_v1()
##inception_v3
model = models.load_inception_v3()
##inception_resnet_v2
#model = models.load_inception_resnet_v2()
##mobilenet
#model = models.load_mobilenet()
##resnet50
#model = models.load_resnet50()



labels = models.load_imagenet_labels()

print('-----LOAD IMAGES-----')
# Load images info
img_paths = utils.load_img_paths()

print('-----PREDICTION-----')
# For each image
i = 0
for img_name, img_path in img_paths.items():
    # Load it
    img = read_image(img_path)
    # Get predicted class
    class_idx = get_pred_class_idx_k(model, labels, img, k=3)
    # Generate File Prediction
    #print(f'IMAGE NAME --> {img_name}')
    i += 1
    with open('out_file_name_IV3.txt', 'a') as f:
        with redirect_stdout(f):
            print(f'{i};{img_name}')
    
    pred_label, pred_prob = get_pred_class_idx_k(model, labels, img, k=3)
    for label, prob in zip(pred_label, pred_prob):
        #print(f'{label}: {prob:0.1%}')
        with open('out_file_probs_IV3.txt', 'a') as f:
            with redirect_stdout(f):
                print(f'{i};{label}: {prob:0.1%}')
    

        
  