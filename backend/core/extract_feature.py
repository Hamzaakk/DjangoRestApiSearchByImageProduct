from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
import tensorflow
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import GlobalMaxPooling2D
import cv2
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle
from sklearn.neighbors import NearestNeighbors


model=ResNet50(weights='imagenet',include_top=False, input_shape=(224,224,3))
model.trainable=False



model=tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

 #take the input
img=cv2.imread("Dataset/1636.jpg")
img=cv2.resize(img, (224,224))

expand_img= np.expand_dims(img,axis=0)

pre_img = preprocess_input(expand_img)

result = model.predict(pre_img).flatten()

normalized = result/norm(result)

print(normalized.shape)

print(result)

def extractfeature(img_path,model):
    img = cv2.imread("Dataset/1636.jpg")
    img=cv2.resize(img, (224,224))
    img= np.array(img)
    expand_img = np.expand_dims(img, axis=0)
    pre_img = preprocess_input(expand_img)
    pre_img = preprocess_input(expand_img)
    result = model.predict(pre_img).flatten()
    normalized = result / norm(result)
    return normalized



filename = []
feature_list = []

for file in os.listdir('Dataset'):
    filename.append(os.path.join('Dataset',file))

for file in tqdm(filename):
    feature_list.append(extractfeature(file,model))



pickle.dump(feature_list, open('../featurevector.pkl', 'wb'))
pickle.dump(filename, open('../filename.pkl', 'wb'))