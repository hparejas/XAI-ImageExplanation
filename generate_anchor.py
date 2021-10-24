import os
import numpy as np

from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
#from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
#from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
#from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

from tensorflow.keras.preprocessing import image

from alibi.explainers import AnchorImage

from settings import IMG_FOLDER, ANCHOR_FOLDER
from utils import saveImageFile, file_is_image, load_img_paths, setup_folder_structure, get_name_without_ext

def transform_img_fn(img_path):
    ##inception_v1 & mobilenet & resnet50
    #img = image.load_img(img_path, target_size=(224, 224))
    ##inception_v3 & inception_resnet_v2
    img = image.load_img(img_path, target_size=(299, 299))

    x = image.img_to_array(img)
    x = preprocess_input(x)
    return x


if __name__ == '__main__':
    setup_folder_structure()

    print('LOADING MODELS AND EXPLAINERS')
    model = InceptionV3(weights='imagenet')
    #model = InceptionResNetV2(weights='imagenet')
    #model = MobileNet(weights='imagenet')
    #model = ResNet50(weights='imagenet')
        
    predict_fn = lambda x: model.predict(x)
    kwargs = {'n_segments': 15, 'compactness': 20, 'sigma': .5}
    
    ##inception_v3 & inception_resnet_v2
    explainer = AnchorImage(predict_fn, image_shape=(299, 299, 3), segmentation_fn='slic', 
                            segmentation_kwargs=kwargs, images_background=None)

    ##inception_v1 & mobilenet & resnet50
    #explainer = AnchorImage(predict_fn, image_shape=(224, 224, 3), segmentation_fn='slic', 
    #                        segmentation_kwargs=kwargs, images_background=None)
    
    print('GENERATING ANCHOR IMAGES')
    count = 1
    img_paths = load_img_paths()
    for img_name, img_path in img_paths.items():
        img = transform_img_fn(img_path)
        np.random.seed(0) # Take note
        explanation = explainer.explain(img, threshold=.95, p_sample=0.5, tau=0.25)
        new_img_name = get_name_without_ext(img_name)
        new_img_path = os.path.join(ANCHOR_FOLDER, f'{new_img_name}.png')
        saveImageFile(explanation.anchor.astype('uint8'), new_img_path) # Convertion from int32 to uint8 needed
        print(f'Generated Images: {count}/{img_paths.__len__()}')
        count += 1
