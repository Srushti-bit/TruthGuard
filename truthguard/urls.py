from django.contrib import admin
from django.urls import path, include
from django.views.generic import RedirectView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/', include('accounts.urls')),
    path('detector/', include('detector.urls')),
    path('dashboard/', include('dashboard.urls')),
    path('', RedirectView.as_view(url='/accounts/login/')),
]