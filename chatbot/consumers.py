from channels.consumer import SyncConsumer,AsyncConsumer
from channels.exceptions import StopConsumer
from channels.generic.websocket import WebsocketConsumer,JsonWebsocketConsumer,AsyncWebsocketConsumer
import time 

# import asyncio
# from asgiref.sync import async_to_sync
import json
from chatbot_application import chatbot as bot
# count = 0
class MyWebsocketConsumer(WebsocketConsumer):
    count = 0
    
    def connect(self):
        print("Websocket Connected...")
        self.accept()
        
    
    def receive(self, text_data):
        print("Client Says :- ",text_data)
        prepro1 = text_data
        if prepro1 != "q":
            reply = bot.insideloop(prepro1)
            print(reply,self.count)
            current_time = time.strftime("%H:%M:%S",time.localtime())
            self.send(text_data=json.dumps({"message": reply,"count":self.count,"current_time": current_time[:5]}))
            
            self.count = self.count+1
            

    def disconnect(self, close_code):
        print("Websocket Disconnected...",close_code)
        
        
        

    