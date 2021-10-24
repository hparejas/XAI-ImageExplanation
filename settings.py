import os

interp_steps = 10
alpha_ratio = (0.5, 0.5)
cmap = 'hot'

# Constants
IMG_FOLDER = os.path.join('.', 'imgs')
IG_FOLDER = os.path.join(IMG_FOLDER, 'ig')
LIME_FOLDER = os.path.join(IMG_FOLDER, 'lime')
XRAI_FOLDER = os.path.join(IMG_FOLDER, 'xrai')
ANCHOR_FOLDER = os.path.join(IMG_FOLDER, 'anchor')