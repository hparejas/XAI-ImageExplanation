import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import explainable_ai_sdk
from explainable_ai_sdk.model import configs
#import tensorflow_hub as hub
#from base64 import b64encode
import base64

from explainable_ai_sdk.metadata.tf.v2 import SavedModelMetadataBuilder
from PIL import Image



import models
import utils
from settings import IMG_FOLDER, XRAI_FOLDER
#from ig import integrated_gradients
from preproc import read_image
from models import get_pred_class_idx


if __name__ == '__main__':
    count = 1

    #############################
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
    ##inception_mobilenet
    #model = models.load_mobilenet()
    ##resnet50
    #model = models.load_resnet50()
    
    labels = models.load_imagenet_labels()

    print('-----LOAD IMAGES-----')
    # Load images info
    img_paths = utils.load_img_paths()

    #############################


    ############# Save the Model
    ##inception_v3
    model_path = 'imagenet_model'
    ##inception_resnet_v2
    #model_path = 'imagenet_model_irv2'
    ##mobilenet
    #model_path = 'imagenet_model_mob'
    ##resnet50
    #model_path = 'imagenet_model_rn50'
    
    def _preprocess(bytes_input):
        decoded = tf.io.decode_jpeg(bytes_input, channels=3)
        decoded = tf.image.convert_image_dtype(decoded, tf.float32)
        ##inception_v3 & inception_resnet_v2
        resized = tf.image.resize(decoded, size=(299, 299))
        ##inception_v1 & mobilenet & resnet50
        #resized = tf.image.resize(decoded, size=(224, 224))
        return resized

    @tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
    def preprocess_fn(bytes_inputs):
        decoded_images = tf.map_fn(_preprocess, bytes_inputs, dtype=tf.float32, back_prop=False)
        return {"numpy_inputs": decoded_images}  # Make note of the key.

    ##inception_v3 & inception_resnet_v2
    m_call = tf.function(model.call).get_concrete_function(
        [tf.TensorSpec(shape=[None, 299, 299, 3], dtype=tf.float32, name="numpy_inputs")])

    ##inception_v1 & mobilenet & resnet50
    #m_call = tf.function(model.call).get_concrete_function(
    #    [tf.TensorSpec(shape=[None, 224, 224, 3], dtype=tf.float32, name="numpy_inputs")])

    
    @tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
    def serving_fn(bytes_inputs):
        images = preprocess_fn(bytes_inputs)
        prob = m_call(**images)
        return prob

    tf.saved_model.save(model, model_path, signatures={
        'serving_default': serving_fn,
        'xai_preprocess': preprocess_fn, # Required for XAI
        'xai_model': m_call # Required for XAI since the default fn takes bytes.
        })

    #############

    ###Create Explanation Metadata
    # Provide the model path and signature to explain.
    md_builder = SavedModelMetadataBuilder(model_path, signature_name='xai_model')

    # Call md_builder.get_metadata() to see inferred inputs and outputs.
    md_builder.set_image_metadata('numpy_inputs')  # Set 'numpy_inputs' input as image input.
    md_builder.save_metadata(model_path)  # Save the metadata in the model path.

    # Reload the Model through Explainable AI SDK
    lm = explainable_ai_sdk.load_model_from_local_path(
        model_path,  # Model path containing explanation metadata JSON.
        explainable_ai_sdk.XraiConfig()  # XRAI config with default step_count=50.
    )

    # Explain an Input
    print('-----INTERPRETATIONS-----')
    # For each image
    for img_name, img_path in img_paths.items():
        # Load it
        #img = read_image(img_path)
        img = img_path

        with open(img, 'rb') as f:
            image_bytes = f.read()
        attributions = lm.explain([{'bytes_inputs': image_bytes}])

        # Attributions contain an attribution per given example.
        #attributions[0].visualize_attributions()
        
        #y.keys() --> dict_keys(['numpy_inputs'])
        y = attributions[0].get_attribution().post_processed_attributions
        
        #list(y.keys()) --> ['numpy_inputs']
        w = list(y.values())
        
        # list(y.values()) --> [{'image_tensor': array([[[75.55601807, 38.95601807, 88.75601807], ... [62.6746994 , 26.0746994 , 75.8746994 ]]]),
        #                     'b64_jpeg': '/9j/4AAQSkZJRgABA
        #w[0].keys() --> dict_keys(['image_tensor', 'b64_jpeg'])
        # w[0].values() --> dict_values([array([[[75.55601807, 38.95601807, 88.75601807], ...  [62.6746994 , 26.0746994 , 75.8746994 ]]]), '/9j/4AAQSkZJRgA
        
        #list(w[0].keys()) --> ['image_tensor', 'b64_jpeg']
 
        z = list(w[0].values())
        #list(w[0].values()) --> [array([[[75.55601807, 38.95601807, 88.75601807], ...          [62.6746994 , 26.0746994 , 75.8746994 ]]]),
        #                       '/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJC
        
        # z[1] --> '/9j/4AAQSkZJRgABAQAAAQAB
        new_img = z[1]


        new_img_name = utils.get_name_without_ext(img_name)
        new_img_path = os.path.join(XRAI_FOLDER, f'{new_img_name}.png')

        # For both Python 2.7 and Python 3.x
        with open(new_img_path, "wb") as fh:
            fh.write(base64.b64decode(new_img))

        #saveImageFile(new_img, new_img_path)
        print(f'Generated Images: {count}/{img_paths.__len__()}')
        count += 1

