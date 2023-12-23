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

from core.models import Product
from django.shortcuts import get_object_or_404
from django.views.decorators.csrf import csrf_exempt

from django.core.files.storage import default_storage

#import some model to help you in models
from django.core.serializers import serialize
import json

# Create your views here.



feature_list=np.array(pickle.load(open("featurevector.pkl", "rb")))
filename=pickle.load(open("filename.pkl", "rb"))



model=ResNet50(weights='imagenet',include_top=False, input_shape=(224,224,3))
model.trainable=False

model=tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])
model.summary()



#to activate crf in order to do a request @csrf_exempt

def extract_feature(img_path, model):
    img = cv2.imread("Dataset/1636.jpg")
    img=cv2.resize(img, (224,224))
    img=np.array(img)
    expand_img=np.expand_dims(img, axis=0)
    pre_img=preprocess_input(expand_img)
    result=model.predict(pre_img).flatten()
    normalized=result/norm(result)
    return normalized


#START
#here we have give to model image and extract feature
'''
normalized=extract_feature("Dataset/1636.jpg", model)
neighbors = NearestNeighbors(n_neighbors=9, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)
distances, indices = neighbors.kneighbors([normalized])

print(indices)

result_list = list()
for file in indices[0][1:6]:
   print(filename[file])
   result_list.append(filename[file])
   
'''
#END EXTRACT FEATURE


#HERE WE LIST SOME PRODUCT TO DISPLAY IN INDEX :
def index(request):
    List_of_products = list()
    List_of_products = filename[1:25]
    data = {'message': ' LIST OF SOME PRODUCT FROM DATASET !','product':List_of_products}
    return JsonResponse(data)

#END OF LIST PRODUCT INDEX



def allProduct(request):
    products = Product.objects.all()
    product_data = serialize('json', products)
    product_list = [item['fields'] for item in json.loads(product_data)]

    data = {'message': 'LIST OF SOME PRODUCT FROM DATASET!', 'product': product_list}

    return JsonResponse(data)




# function give image and return result with model
@csrf_exempt
def getResult(image_path, model):
    result_list = list()
    #normalize image input
    normalized = extract_feature("Dataset/1636.jpg", model)
    neighbors = NearestNeighbors(n_neighbors=9, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([normalized])

    print(indices)

    result_list = list()
    for file in indices[0][1:6]:
        print(filename[file])
        result_list.append(filename[file])

    return result_list
# end

@csrf_exempt
def getResult2(image_path, model):
    # Normalize image input
    normalized = extract_feature(image_path, model)

    neighbors = NearestNeighbors(n_neighbors=9, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([normalized])

    result_list = []
    for file in indices[0][1:6]:
        result_list.append(filename[file])

    return result_list



@csrf_exempt
def extract_feature(request):
    result_products = list()
    if request.method == 'POST':
        try:
            # Assuming the image is sent as form data with key 'image'
            img = request.FILES['image'].read()
            img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
            result_products = getResult(img, model)  # Use result_products here
            data = {'message': 'Image feature extracted successfully', 'result_list': result_products}
            return JsonResponse(data)

        except Exception as e:
            data = {'error': str(e)}
            return JsonResponse(data, status=400)

    else:
        data = {'error': 'Invalid request method. This page is designed for POST requests.'}
        return JsonResponse(data, status=400)


#get product by id

def get_product_by_id(request, product_id):
    product = get_object_or_404(Product, id=product_id)
    product_data = serialize('json', [product])
    product_dict = json.loads(product_data)[0]['fields']

    data = {'message': f'Details for Product ID {product_id}', 'product': product_dict}

    return JsonResponse(data)


@csrf_exempt  # Use this decorator for simplicity in this example; consider a better solution in production
@csrf_exempt
@csrf_exempt
def upload_image(request):
    result_list = []

    if request.method == 'POST' and 'image' in request.FILES:
        try:
            uploaded_image = request.FILES['image']

            # Save the image to the 'uploads' folder in your media directory
            file_path = default_storage.save(f'uploads/{uploaded_image.name}', uploaded_image)
            print(file_path)

            # Get results using getResult
            img = cv2.imread(f'media/{file_path}')
            img = cv2.resize(img, (224, 224))
            img = np.array(img)
            expand_img = np.expand_dims(img, axis=0)
            pre_img = preprocess_input(expand_img)
            result = model.predict(pre_img).flatten()
            normalized = result / norm(result)

            neighbors = NearestNeighbors(n_neighbors=9, algorithm='brute', metric='euclidean')
            neighbors.fit(feature_list)
            distances, indices = neighbors.kneighbors([normalized])
            print(indices)
            for file in indices[0][1:7]:
                #print(filename[file])
                result_list.append(filename[file])
            data = {'message': result_list}
            return JsonResponse(data)





        except Exception as e:
            data = {'error': str(e)}
            return JsonResponse(data, status=400)

    else:
        data = {'error': 'Invalid request method or no image provided', }
        return JsonResponse(data, status=400)