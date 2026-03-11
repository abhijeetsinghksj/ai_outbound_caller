from django.urls import path
from . import views
urlpatterns = [
    path("answer/",  views.answer_call,       name="answer_call"),
    path("respond/", views.respond_to_speech, name="respond_to_speech"),
    path("status/",  views.call_status,       name="call_status"),
]
