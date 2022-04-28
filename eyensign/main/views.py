from django.shortcuts import render
from .forms import ImageForm
from .sign_predict import predict

# Create your views here.


def main(request):
    if request.method =="POST":        
        form = ImageForm(request.POST, request.FILES)
        try:
            if form.is_valid():
                form.save()
        except:
            pass

        img_path ="./static/" + str(request.FILES["image_field"]).replace(" ", "_")
        
        result = predict(img_path)
        
        return render(request, "main.html", {"image_form": None,  "result":result})
    else:
        image_form = ImageForm()

        return render(request, "main.html", {"image_form": image_form, "result":None})
