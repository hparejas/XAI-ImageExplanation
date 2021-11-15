# XAI-ImageExplanation

IMAGE CLASSIFICATION
This application predicts the top 3 classification of an image using Inception_V3 model, the first prediction considers an accuracy between 55% and 75%, and the accuracy for the second prediction is greater to 30%. The data model used in the experiment is Visual Genome.

INTERPRETATION IMAGES GENERATOR "XAI"
This tool is used to generate interpretation images for any arbitrary dataset of images. The techniques used for interpretations are Integrated Gradients (IG), LIME, XRAI and ANCHOR.

REQUIREMENTS
This project used Python 3.8.5 but should work with any Python 3.6+ installation.

Also, the project uses several scientific Python libraries, like Tensorflow and Numpy, to name a few.

Virtual environment
Although the libraries needed for the project can be installed globally, is it highly recommended that virtual environments are used. We used CONDA virtual environments to generate these environments.

Install dependencies
The project uses many libraries, with specific versions so they do not cause errors.

Image generation dependencies
If you want to use the generation functionalities of the project, use the following command

$ pip install -r requirements-gen.txt
If, by any reason, an error happens, installation can be manual:

EXECUTION AND USAGE
Image Generation
The project uses 4 scripts to generate interpretation images. The files used to generate images are:

generate_ig.py (for Integrated Gradients technique)
generate_lime.py (for LIME technique)
generate_xrai.py (for XRAI technique)
generate_anchor.py (for ANCHOR technique)
To generate images, simply type the folowing in a terminal

$ python [script_name]
For example, to generate LIME images, type

$ python generate_lime.py
