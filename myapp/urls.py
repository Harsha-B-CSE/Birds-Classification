from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from myapp.views import (
    predict_view,
    index,
    about,
    aboutus,
    login_view,
    signup_view,
    logout_view,
    species_profile_view,
    history_view,
    history_api_view,
)

urlpatterns = [
    path("", predict_view, name="home"),
    path("predict/", predict_view, name="predict"),
    path("about/", about, name="about"),
    path("aboutus/", aboutus, name="aboutus"),
    path("index/", index, name="index"),
    path("api/species/", species_profile_view, name="species_profile"),
    path("api/history/", history_api_view, name="history_api"),
    path("history/", history_view, name="history"),
    # Legacy authentication paths
    path("login/", login_view, name="login"),
    path("signup/", signup_view, name="signup"),
    path("logout/", logout_view, name="logout"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
