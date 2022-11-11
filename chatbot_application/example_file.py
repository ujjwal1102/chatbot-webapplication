
# def e_file():
#     que = str(input("Enter question"))
#     return que

import chatbot

def loopModel():
    prepro1 = ""
    while prepro1 != 'q':
        ques  = str(input("Enter question"))
        # ques= ef.e_file()
        # ques = message
        print(ques)
        reply = chatbot.insideloop(ques)
        
        

    

loopModel()




# x = loopModel()
# x

# def chat_reply_message(request):
#     from chatbot_application import chatbot
#     run_chatbot = chatbot.ChatBot()
#     msg = {'text' : run_chatbot.reply}
#     return render(request,msg)

