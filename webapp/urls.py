from django.contrib import admin
from django.urls import path, include
from users import views as user_view
# from django.contrib.auth import views as auth

 
urlpatterns = [
 
    path('admin/', admin.site.urls),
 
    ##### user related path##########################
    path('', include('users.urls')),
    

    
 
]