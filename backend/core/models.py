from django.db import models

# Create your models here.

# myapp/models.py

from django.db import models

class Product(models.Model):
    name = models.CharField(max_length=30)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    category = models.CharField(max_length=30)
    rating = models.CharField(max_length=30)
    description = models.TextField()
    quantity = models.IntegerField()
    image_path = models.TextField()
    active = models.BooleanField(default=True)
