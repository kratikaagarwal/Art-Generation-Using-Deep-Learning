import cv2     # for capturing videos
import math   # for mathematical operations
import matplotlib.pyplot as plt    # for plotting the images
%matplotlib inline
import pandas as pd
from keras.preprocessing import image   # for preprocessing the images
import numpy as np    # for mathematical operations
from keras.utils import np_utils, to_categorical
from skimage.transform import resize   # for resizing images
import os
import tqdm
from tqdm import tqdm,tqdm_pandas
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras import applications,models,losses,optimizers
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_last')

from glob import glob

from numpy import genfromtxt

import tensorflow as tf

from keras import applications

from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D,LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.metrics import categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import Input, Dense, Flatten, Dropout, Activation, Lambda, Permute, Reshape
import keras.backend as K
import scipy as sp
from scipy.spatial import distance
from PIL import Image
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from hyperas.distributions import choice, uniform
import hyperopt
from sklearn.model_selection import train_test_split
from keras.utils import Sequence, to_categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import *
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from keras.layers import (Convolution1D, Dense, Dropout, GlobalAveragePooling1D, 
                          GlobalMaxPool1D, Input, MaxPool1D, concatenate)
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from os import listdir
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding,LSTM,LSTMCell
from keras.layers.merge import add
from keras.models import Model
from keras.utils import plot_model
from nltk.translate.bleu_score import corpus_bleu
from keras.models import load_model
from skimage.io import imread, imshow, imread_collection, concatenate_images
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from __future__ import print_function, division
from builtins import range, input
from datetime import datetime

tf.compat.v1.disable_eager_execution()

def load_preprocess_img(p,shape = None):
    Img = image.load_img(p, target_size=shape)
    X = image.img_to_array(Img)
    X = np.expand_dims(X,axis=0)    
    X = preprocess_input(X)
    return X

def preprocess_img(frame,shape = None):
    X = np.expand_dims(frame,axis=0)    
    X = preprocess_input(X.astype(('float64')))
    return X
    

#Loading style image
    
style_img = load_preprocess_img(p = '/images/style.jpg', shape=(224,224))
batch_shape = style_img.shape
shape = style_img.shape[1:]
shape

#Content model define
def vgg_avg_pooling(shape):
    vgg = VGG16(input_shape=shape,weights='imagenet',include_top=False)
    model = Sequential()
    for layer in vgg.layers:
        if layer.__class__ == MaxPooling2D:
        # replace it with average pooling    
            model.add(AveragePooling2D())
        else:
            model.add(layer)
    return model   

def vgg_cutoff(shape,num_conv):
    if num_conv<1|num_conv>13:
        print('Error layer must be with in [1,13]')
    model = vgg_avg_pooling(shape)
    new_model = Sequential()
    n=0
    for layer in model.layers:
        new_model.add(layer)
        if layer.__class__ == Conv2D:
            n+=1
        if n >= num_conv:
            break
    return new_model

#Style loss comutation graph

def gram_matrix(img):
    # input is (H, W, C) (C = # feature maps)
    # we first need to convert it to (C, H*W)
    X = K.batch_flatten(K.permute_dimensions(img,(2,0,1)))
    # now, calculate the gram matrix
    # gram = XX^T / N
    gram_mat = K.dot(X,K.transpose(X))/img.get_shape().num_elements()
    return gram_mat 

def style_loss(y,t):
    return K.mean(K.square(gram_matrix(y)-gram_matrix(t)))

def unpreprocess(img):
    img[..., 0] += 103.939
    img[..., 1] += 116.779
    img[..., 2] += 126.68
    img = img[..., ::-1]
    return img

def scale(x):
    x = x-x.min()
    x=x/x.max()
    return x


#Style image load and VGG model load.
path = '/images/style.jpg'
img = image.load_img(path)
x = image.img_to_array(img)
x = np.expand_dims(x,axis=0)    
x = preprocess_input(x)
    #shape
batch_shape = x.shape
shape = x.shape[1:]
vgg = vgg_avg_pooling(shape)

#Define multi-output model
symb_conv_outputs = [layer.get_output_at(1) for layer in vgg.layers if layer.name.endswith('conv1')]
multi_output_model = Model(vgg.input, symb_conv_outputs)
symb_layer_out = [K.variable(y) for y in multi_output_model.predict(x)]

#Conv layer weight matrix
weights = [0.2,0.4,0.3,0.5,0.2]    
loss=0
#Total style loss
for symb,actual,w in zip(symb_conv_outputs,symb_layer_out,weights):
    loss += w * style_loss(symb[0],actual[0])
    
#gradients which are needed by the optimizer    
grad = K.gradients(loss,multi_output_model.input)
get_loss_grad = K.function(inputs=[multi_output_model.input], outputs=[loss] + grad)

#Scipy's minimizer function(fmin_l_bfgs_b) allows us to pass back function value f(x) and 
#its gradient f'(x), which we calculated in earlier step. 
#However, we need to unroll the input to minimizer function in1-D array format and both loss and gradient must be np.float64.

def get_loss_grad_wrapper(x_vec):
    l,g = get_loss_grad([x_vec.reshape(*batch_shape)])
    return l.astype(np.float64), g.flatten().astype(np.float64)    

#Function to minimize loss
import scipy
def min_loss(fn,epochs,batch_shape):
    t0 = datetime.now()
    losses = []
    x = np.random.randn(np.prod(batch_shape))
    for i in range(epochs):
        x, l, _ = scipy.optimize.fmin_l_bfgs_b(func=fn,x0=x,maxfun=20)
    # bounds=[[-127, 127]]*len(x.flatten())
    #x = np.clip(x, -127, 127)
    # print("min:", x.min(), "max:", x.max())
        print("iter=%s, loss=%s" % (i, l))
        losses.append(l)
    print("duration:", datetime.now() - t0)
    plt.plot(losses)
    plt.show()

    newimg = x.reshape(*batch_shape)
    final_img = unpreprocess(newimg)
    return final_img[0]    

import cv2
import numpy as np
 
# Create a VideoCapture object
batch_shape = style_img.shape
shape = style_img.shape[1:]
cap = cv2.VideoCapture('/output/input.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4)) 
# Check if camera opened successfully
if (cap.isOpened() == False): 
    print("Unable to read camera feed")
 
# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
 
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('/output/outpy.mp4',cv2.VideoWriter_fourcc(*'MJPG'),20.0,(224,224))
count = 0 
while(True):
    ret, frame = cap.read()
    frame = cv2.resize(frame,(224,224))
    X = preprocess_img(frame)
    vgg = vgg_avg_pooling(shape=shape)
    content_model = Model(vgg.input,vgg.layers[13].get_output_at(0))
    content_target = content_model.predict(X)
    symb_conv_outputs = [layer.get_output_at(1) for layer in vgg.layers if layer.name.endswith('conv1')]
    multi_output_model = Model(vgg.input, symb_conv_outputs)
    symb_layer_out = [K.variable(y) for y in multi_output_model.predict(style_img)]
    weights = [0.2,0.4,0.3,0.5,0.2]
    
    loss=K.mean(K.square(content_model.output-content_target)) *4 
    for symb,actual,w in zip(symb_conv_outputs,symb_layer_out,weights):
        loss += 0.03 * w * style_loss(symb[0],actual[0])
    
    grad = K.gradients(loss,vgg.input)
    get_loss_grad = K.function(inputs=[vgg.input], outputs=[loss] + grad)
    def get_loss_grad_wrapper(x_vec):
        l,g = get_loss_grad([x_vec.reshape(*batch_shape)])
        return l.astype(np.float64), g.flatten().astype(np.float64)
    final_img = min_loss(fn=get_loss_grad_wrapper,epochs=30,batch_shape=batch_shape)
    plt.imshow(scale(final_img))
    plt.show()
    
    
    #cv2.imshow('output',frame)
    #out.write(final_img)
    filename ="/output/style_images/frame%d.jpg" % count;count+=1 
    cv2.imwrite(filename, final_img)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
        #break
     
    # Write the frame into the file 'output.avi'
        
 
    # Display the resulting frame    
        
  # Break the loop
    #if k == 27:
    #    break 
 
# When everything done, release the video capture and video write objects
cap.release()
out.release()
cv2.destroyAllWindows()


image_folder = '/output/style_images'
video_name = '/output/video.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()    