# import re
from django.urls import re_path

from . import consumers

websocket_urlpatterns =[
    re_path('ws/wsc/',consumers.MyWebsocketConsumer.as_asgi()),
    # re_path('ws/awsc/',consumers.MyAsyncConsumer.as_asgi()),
]