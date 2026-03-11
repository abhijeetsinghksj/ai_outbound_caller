from django.urls import path, include
urlpatterns = [
    path("calls/", include("calls.urls")),
]
