from django.conf.urls import url
from predsuccess.views import predictSuccess


urlpatterns = [
    url('', predictSuccess, name='predict'),
]
