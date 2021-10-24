import os
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
from PIL import Image as im

from settings import *
from preproc import read_image
from models import get_pred_class_idx

def setup_folder_structure():
    if not os.path.isdir(IG_FOLDER):
        os.mkdir(IG_FOLDER)
    if not os.path.isdir(LIME_FOLDER):
        os.mkdir(LIME_FOLDER)
    if not os.path.isdir(XRAI_FOLDER):
        os.mkdir(XRAI_FOLDER)
    if not os.path.isdir(ANCHOR_FOLDER):
        os.mkdir(ANCHOR_FOLDER)

def get_name_without_ext(filename):
    return '.'.join(filename.split('.')[:-1])

def file_is_image(img_name):
    return img_name.split('.')[-1]=='jpeg' or img_name.split('.')[-1]=='jpg'

def load_img_paths():
    img_paths = {img_name: os.path.join(IMG_FOLDER, img_name) for img_name in os.listdir(IMG_FOLDER) if file_is_image(img_name)}
    return img_paths

def load_img_tensors(img_paths):
    img_tensors = {name: read_image(img_path) for (name, img_path) in img_paths.items()}
    return img_tensors

def get_imgs_classes(model, labels, img_tensors):
    img_preds = {img_name: get_pred_class_idx(model, labels, img) for (img_name, img) in img_tensors.items()}
    return img_preds

def convertMatrixToRGBA(img, cmap=cmap):
    norm = Normalize(vmin=np.min(img), vmax=np.max(img))
    mapper = ScalarMappable(norm=norm, cmap=cmap)
    return mapper.to_rgba(img, bytes=True, norm=True)

def merge_images(img, mask, img_alpha=alpha_ratio[0], mask_alpha=alpha_ratio[1]):
    return (img*img_alpha + mask*mask_alpha).astype(np.uint8)

def saveImageFile(img, img_name):
    im.fromarray(img).save(img_name)