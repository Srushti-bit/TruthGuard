from django.urls import path
from . import views

app_name = 'detector'

urlpatterns = [
    path('', views.submit_news, name='submit'),
    path('history/', views.history_view, name='history'),
    path('delete/<int:submission_id>/', views.delete_submission, name='delete_submission'),
]