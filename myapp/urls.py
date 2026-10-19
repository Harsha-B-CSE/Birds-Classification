from django.contrib import admin
from django.urls import path
from myapp.views import*

from django.urls import path
from myapp.views import *

urlpatterns = [
    path('admin/', admin.site.urls), 
    path("login/", login_view, name="login"),  # Change from "" to "login/"
    path("signup/", signup_view, name="signup"),
    path("logout/", logout_view, name="logout"),
    #path("index/", index, name="index"),
    path("about/", about, name="about"),
    path("aboutus/", aboutus, name="aboutus"),
    path("predict/", predict_view, name="predict"),
    path("", index, name="home"),
    
]
from django.conf import settings
from django.conf.urls.static import static

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)