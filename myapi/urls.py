from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('get-response/<str:user_message>/', views.get_response, name='get_response'),
    path('make-appointment/<str:user_message>/', views.make_appointment, name='make_appointment'),
]
