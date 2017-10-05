"""
Prepare the web service definition by authoring init() and run() functions. 
Test the functions before deploying the web service.
"""

# To generate the schema file, simply execute this scoring script. Make sure that you are using Azure ML Python environment.
# Running this file creates a service-schema.json file under the current working directory that contains the schema 
# of the web service input. It also test the scoring script before creating the web service.
# cd C:\Users\<user-name>\AppData\Local\amlworkbench\Python\python score.py

#Here is the CLI command to create a realtime scoring web service

# Create realtime service
#az ml env setup -g env4entityextractorrg -n env4entityextractor --cluster -z 5 -l eastus2

# Set up AML environment and compute with ACS
#az ml env set --cluster-name env4entityextractor --resource-group env4entityextractorrg

#C:\dl4nlp\models>az ml service create realtime -n extract-biomedical-entities -f score.py -m lstm_bidirectional_model.h5 -s service-schema.json -r python -d resources.pkl -d DataReader.py -d EntityExtractor.py -c scoring_conda_dependencies.yml

#Here is the CLI command to run Kubernetes
#C:\Users\<user-name>\bin\kubectl.exe proxy --kubeconfig C:\Users\hacker\.kube\config


import os
import sys
import json
import numpy as np
#import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense
from azureml.api.schema.dataTypes import DataTypes
from azureml.api.schema.sampleDefinition import SampleDefinition
from azureml.api.realtime.services import generate_schema

img_width, img_height = 28, 28

# init loads the model (global)
def init():
    """ Initialize and load the model
    """
    global model

    # define architecture of the model.  same architecture used in the train step
    model = Sequential()

    model.add(Conv2D(16, (5, 5), input_shape=(img_width, img_height, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())    # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(1000))
    model.add(Activation('relu'))

    model.add(Dense(10))
    model.add(Activation('softmax'))
    # finished model definition

    try:
        # this model was downloaded and copied to the this location on the execution
        # target after the training step prior to running this script
        model_file_path = os.path.join("/tmp", "mnistneuralnet.h5")
        
        #load the model
        print("Loading trained neural net {}".format(model_file_path))
        model.load_weights(model_file_path)
    except:
        print("can't load the neural network model")
        pass

 
# run takes an input numpy array and performs prediction   
def run(input_array):
    """ Classify the input using the loaded model
    """
    
    try:
        # model.predict returns something like [[0,1,0,0,0,0,0,0,0,0]], so we take the 0th element
        prediction = model.predict(input_array)[0]

        best_class = ''
        best_conf = -1
        for n in [0,1,2,3,4,5,6,7,8,9]:
            if (prediction[n] > best_conf):
                best_class = str(n)
                best_conf = prediction[n]

        return best_class
    except Exception as e:
        return (str(e))

   

def main(): 
    import cv2

    # if image file path not provided as argument, then pick a file from the test sample
    if len(sys.argv) > 1:
        path_to_image = sys.argv[1]
    else:
        path_to_image = "/tmp/data/mnist_png/testing/8/61.png"

    
    img = cv2.imread(path_to_image)
    img = cv2.resize(img, (img_width, img_height))

    # read in image and resize to 28x28
    #img = Image.open(path_to_image)
    #img = img.resize((img_width, img_height))

    # convert image to numpy array
    input_array = np.array(img).reshape((img_width,img_height,3))
    input_array = np.expand_dims(input_array, axis=0)

    # Test the output of the functions
    init()  
    
    print("Result: " + run(input_array))

    inputs = {"input_array": SampleDefinition(DataTypes.NUMPY, input_array)}

    # create the outputs folder
    os.makedirs('./outputs', exist_ok=True)

    #Generate the schema
    generate_schema(run_func=run, inputs=inputs, filepath='./outputs/service-schema.json')
    print("Schema generated")


if __name__ == "__main__":
    main()
