import os
from tensorflow import zeros

import models
import utils
from settings import IG_FOLDER, cmap
from ig import integrated_gradients
from preproc import read_image
from models import get_pred_class_idx

##inception_v1 & mobilenet & resnet50
#baseline = zeros(shape=(224,224,3))
##inception_v3
baseline = zeros(shape=(299,299,3))


print('-----SETUP-----')
# Setup
utils.setup_folder_structure()

print('-----LOAD MODELS-----')
# Load models
##inception_v1
#model = models.load_inception_v1()
##inception_v3
model = models.load_inception_v3()
##inception_v3
#model = models.load_inception_resnet_v2()
##MobileNet
#model = models.load_mobilenet()
##ResNet50
#model = models.load_resnet50()

labels = models.load_imagenet_labels()



print('-----LOAD IMAGES-----')
# Load images info
img_paths = utils.load_img_paths()

print('-----INTERPRETATIONS-----')
# For each image
for img_name, img_path in img_paths.items():
    # Load it
    img = read_image(img_path)
    # Get predicted class
    class_idx = get_pred_class_idx(model, labels, img)
    # Generate IG's
    mask = integrated_gradients(model=model,
                                baseline=baseline,
                                image=img,
                                target_class_idx=class_idx,
                                m_steps=10)
    # Merge images
    mask_img = utils.convertMatrixToRGBA(mask.numpy(), cmap)
    orig_img = utils.convertMatrixToRGBA(img.numpy())
    merged_img = utils.merge_images(orig_img, mask_img)
    # Save images
    new_name = utils.get_name_without_ext(img_name)
    new_ig_img_url = os.path.join(IG_FOLDER, f'{new_name}.png')
    utils.saveImageFile(merged_img, new_ig_img_url)