from django.contrib import admin
from django.urls import path,include
from frontend import views

urlpatterns = [
    path('',views.home,name='home'),
    path('signup/', views.register_view, name='signup'),
    path('login/', views.login_view, name='login'),
    path('logout',views.logout_view,name='logout'),
    path('home/', views.home, name='home'),
    path('history/', views.history, name='history'),
    path('virtual-assistant/', views.virtual_assistant, name='virtual_assistant'),
    path('dynamic-pricing/', views.dynamic_pricing, name='dynamic_pricing'),
    path('travel_safety/', views.travel_safety, name='travel_safety'),
    path('travel_safety/', views.travel_safety, name='hospital_locator'),
    path('tourist_guide/', views.tourist_guide, name='touristguide'),
    path('get-trip-details/', views.get_trip_details, name='gettripdetails'),
    path('safety_tips/', views.safety_tips, name='safety_tips'),
    path('contact/', views.contact, name='contact'),
    
    # Add other paths as needed
]
