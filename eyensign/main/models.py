from django.db import models

# Create your models here.

class ImageModel(models.Model):
    image_field = models.ImageField(upload_to="static")