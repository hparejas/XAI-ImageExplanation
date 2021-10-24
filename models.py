import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

def load_inception_v1():
    model = tf.keras.Sequential([
        hub.KerasLayer(
            name='inception_v1',
            handle='https://tfhub.dev/google/imagenet/inception_v1/classification/4',
            trainable=False
        )
    ])
    model.build([None, 224, 224, 3])
    return model


def load_inception_v3():
    m = tf.keras.Sequential([
        hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_v3/classification/5")
    ])
    m.build([None, 299, 299, 3])  # Batch input shape.
    return m


def load_inception_resnet_v2():
    m = tf.keras.Sequential([
        hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_resnet_v2/classification/5")
    ])
    m.build([None, 299, 299, 3])  # Batch input shape.  
    
    '''
    tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
        include_top=True, weights='imagenet', input_tensor=None,
        input_shape=None, pooling=None, classes=1000,
        classifier_activation='softmax', **kwargs
    )    

    tf.keras.applications.inception_resnet_v2.preprocess_input(
        x, data_format=None
    )
    '''
    return m


def load_mobilenet():
    m = tf.keras.Sequential([
        hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/5")
    ])
    m.build([None, 224, 224, 3])  # Batch input shape.
    
    '''
    tf.keras.applications.mobilenet.MobileNet(
        input_shape=None, alpha=1.0, depth_multiplier=1, dropout=0.001,
        include_top=True, weights='imagenet', input_tensor=None, pooling=None,
        classes=1000, classifier_activation='softmax', **kwargs
    )    

    tf.keras.applications.mobilenet.preprocess_input(
        x, data_format=None
    )
    '''
    return m


def load_resnet50():
    m = tf.keras.Sequential([
        hub.KerasLayer("https://tfhub.dev/tensorflow/resnet_50/classification/1")
    ])
    m.build([None, 224, 224, 3])  # Batch input shape.    
    
    '''
    tf.keras.applications.resnet50.ResNet50(
        include_top=True, weights='imagenet', input_tensor=None,
        input_shape=None, pooling=None, classes=1000, **kwargs
    )
    
    tf.keras.applications.resnet50.preprocess_input(
        x, data_format=None
    )
    '''
    return m


def load_imagenet_labels(url='https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'):
    labels_file = tf.keras.utils.get_file('ImageNetLabels.txt', url)
    with open(labels_file) as reader:
        f = reader.read()
        labels = f.splitlines()
    return np.array(labels)

def get_pred_class_idx(model, labels, image):
    image_batch = tf.expand_dims(image, 0)
    predictions = model(image_batch)
    probs = tf.nn.softmax(predictions, axis=-1)
    top_probs, top_idxs = tf.math.top_k(input=probs, k=1)
    top_label = labels[tuple(top_idxs[0])]
    return top_idxs[0].numpy()[0]


def get_pred_class_idx_k(model, labels, image, k):
    image_batch = tf.expand_dims(image, 0)
    predictions = model(image_batch)
    probs = tf.nn.softmax(predictions, axis=-1)
    top_probs, top_idxs = tf.math.top_k(input=probs, k=k)
    top_labels = labels[tuple(top_idxs)]
    return top_labels, top_probs[0]
