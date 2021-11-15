import os
from tensorflow import zeros

from contextlib import redirect_stdout

import models
import utils
from settings import IMG_FOLDER, NEW_FOLDER
#from ig import integrated_gradients
from preproc import read_image
from models import get_pred_class_idx_k

from utils import saveImageFile, load_img_paths, setup_folder_structure, get_name_without_ext

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
    
    '''
    with open('out_file_name_IV3.txt', 'a') as f:
        with redirect_stdout(f):
            print(f'{i};{img_name}')
    
    '''
    pred_label, pred_prob = get_pred_class_idx_k(model, labels, img, k=3)
    
    cont = 0
    flag1 = False
    flag2 = False
    flag3 = False
    prob1 = 0
    prob2 = 0
    prob3 = 0
    label1 = ""
    label2 = ""
    label3 = ""
    
    
    for label, prob in zip(pred_label, pred_prob):
        #print(f'{label}: {prob:0.1%}')
        cont +=1
        #print(prob)
        if cont == 1 and (prob >= 0.55 and prob <= 0.75):
            flag1 = True
            prob1 = prob
            label1 = label
        elif cont == 2 and prob > 0.3:
            flag2 = True
            prob2 = prob
            label2 = label
        elif cont == 3:
            flag3 = True
            prob3 = prob
            label3 = label
        
        if flag1 and flag2 and flag3:
            #copy image in another folder & write
            with open('out_file_probs_IV3.txt', 'a') as f:
                with redirect_stdout(f):
                    print(f'{i};{label1}: {prob1:0.1%}')
                    print(f'{i};{label2}: {prob2:0.1%}')
                    print(f'{i};{label3}: {prob3:0.1%}')
    
    if flag1 and flag2:
        with open('out_file_name_IV3.txt', 'a') as f:
            with redirect_stdout(f):
                print(f'{i};{img_name}')

        #Save Image
        new_img = utils.convertMatrixToRGBA(img.numpy())
        new_img_name = get_name_without_ext(img_name)
        new_img_path = os.path.join(NEW_FOLDER, f'{new_img_name}.png')
        saveImageFile(new_img, new_img_path)
 