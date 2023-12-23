from django.urls import path,include
from .views import index , extract_feature , allProduct , get_product_by_id,upload_image

urlpatterns = [
    path('',allProduct,name='allProduct'),
    path('product/<int:product_id>/', get_product_by_id, name='product_by_id'),
    path('extractfeature/', extract_feature, name='extract_feature'),
    path('upload/', upload_image, name='upload_image'),

]