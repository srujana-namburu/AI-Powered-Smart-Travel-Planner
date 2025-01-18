from django.http import HttpResponse,JsonResponse
# Import necessary modules and models
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from .models import *

# Define a view function for the home page
def login_view(request):
    if request.method == "POST":
        email = request.POST.get('email')
        password = request.POST.get('password')
        user = authenticate(request, username=email, password=password)
        if user is not None:
            login(request, user)
            return redirect('home')  # Redirect to the home page
        else:
            messages.error(request, "Invalid email or password.")
    return render(request, 'frontend/login.html')



# Define a view function for the registration page
def register_view(request):
    # Check if the HTTP request method is POST (form submission)
    if request.method == 'POST':
        email = request.POST['mailid']
        password = request.POST['password']

        # Check if a user with the provided username already exists
        try:
            user = User.objects.create_user(username=email,password=password)
            user.save()
            login(request,user)
            return redirect('home')
        except:
            messages.error(request,"Username already exists")

    
    # Render the registration page template (GET request)
    return render(request, 'frontend/signup.html')

def logout_view(request):
    logout(request)
    return redirect('login')


def Signin(request):
    return render(request,'frontend/home.html')

def Register(request):
    return render(request,'frontend/home.html')

def login(request):
    return render(request, 'frontend/login.html') 

def signup(request):
    return render(request, 'frontend/signup.html') 

def home(request):
    return render(request, 'frontend/home.html')

def history(request):
    return render(request, 'frontend/history.html')

def virtual_assistant(request):
    return render(request, 'frontend/virtual_assistant.html')

def dynamic_pricing(request):
    return render(request, 'frontend/dynamic_pricing.html')

def travel_safety(request):
    return render(request, 'frontend/travel_safety.html')

def hospital_locator(request):
    return render(request, 'frontend/travel_safety.html')

def tourist_guide(request):
    return render(request, 'frontend/tourist_guide.html')

def get_trip_details(request):
    return render(request, 'frontend/get_trip_details.html')

def safety_tips(request):
    return render(request, 'safety_tips.html')

def contact(request):
    return render(request, 'contact.html')
