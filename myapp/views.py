import os
import numpy as np
import pandas as pd
import tensorflow as tf
from django.shortcuts import render, redirect, get_object_or_404

from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from tensorflow.keras.preprocessing import image
from django.http import JsonResponse
from django.shortcuts import render
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os


from django.shortcuts import render
from django.http import JsonResponse
from PIL import Image
import torch
import torchvision.transforms as transforms
import timm

from django.shortcuts import render
from django.http import JsonResponse
from PIL import Image
import torch
import torchvision.transforms as transforms
import timm
import json
import torch.nn as nn
import os

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load cleaned class names from the JSON file
class_names_path = os.path.join("D:\\Bird_class\\myapp\\cleaned_classes.json")
try:
    with open(class_names_path, "r") as file:
        class_names = json.load(file)
except Exception as e:
    print(f"Error loading class names: {e}")
    class_names = []

# Custom model with Swin Transformer
class CUBModel(nn.Module):
    def __init__(self, num_classes=200, pretrained=True):
        super(CUBModel, self).__init__()
        self.model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=pretrained, num_classes=num_classes)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.dropout(self.model(x))
        return x

# Load the trained model
model_path = os.path.join("D:\\Bird_class\\myapp\\swin_model_best.pth")
model = CUBModel(num_classes=len(class_names)).to(device)

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_view(request):
    if request.method == "POST" and request.FILES.get("image"):
        try:
            # Load and preprocess the image
            image = Image.open(request.FILES["image"]).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)

            # Perform prediction
            with torch.no_grad():
                output = model(image_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                top_prob, top_class = torch.topk(probabilities, 1)

            predicted_class = class_names[top_class.item()]
            predicted_probability = top_prob.item()

            return JsonResponse({"class": predicted_class, "probability": f"{predicted_probability:.4f}"})

        except Exception as e:
            print(f"Prediction error: {e}")
            return JsonResponse({"error": "Prediction failed. Please try again."})

    return render(request, "myapp/predict.html")






# Login View
def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')  
        password = request.POST.get('password')  

        if username and password:
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('index')
            else:
                messages.error(request, "Invalid credentials")
        else:
            messages.error(request, "Both fields are required")

    return render(request, 'myapp/login.html')  # Ensure correct template path

# Signup View
def signup_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password1 = request.POST['password1']
        password2 = request.POST['password2']

        if password1 != password2:
            messages.error(request, "Passwords do not match!")
            return redirect('signup')

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already taken!")
            return redirect('signup')

        user = User.objects.create_user(username=username, email=email, password=password1)
        user.save()
        messages.success(request, "Account created successfully! Please login.")
        return redirect('login')

    return render(request, 'myapp/signup.html')

# Logout View
def logout_view(request):
    logout(request)
    return redirect('login')

# Index View
@login_required
def index(request):
    return render(request, 'myapp/index.html')


# About Pages
def about(request):
    return render(request, 'myapp/about.html')

def aboutus(request):
    return render(request, 'myapp/aboutus.html')
