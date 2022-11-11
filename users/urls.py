from multiprocessing import AuthenticationError
from django.urls import path, include
from django.conf import settings
from django.contrib.auth.views import LoginView,LogoutView
# from regex import template
from . import views
from .forms import LoginForm
# from django.contrib.auth import views as  auth_views
# from django.conf.urls.static import static
 
urlpatterns = [
         path('', views.index, name ='index'),
         path('login/',LoginView.as_view(template_name='login.html',authentication_form=LoginForm),name='login'),
         path('register/',views.register,name='signup'),
         path('chatbot/',views.chatbot,name='chatbot'),
         path('about/',views.about,name="about"),
        #  path('', include('chatbot.urls')),
         path('logout/',LogoutView.as_view(),name='logout'),
]