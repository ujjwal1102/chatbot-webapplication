from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm,UserCreationForm

# from generative_basd_chatbot.chatbot import ChatBot
from .forms import UserRegisterForm
from django.core.mail import send_mail
from django.core.mail import EmailMultiAlternatives
from django.template.loader import get_template
from django.template import Context
# from .forms import UserRegisterForm

  
#################### index#######################################
def index(request):
    return render(request, 'homepage.html', {'page_title':'Homepage'})

########### register here #####################################
def register(request):
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            # username = form.cleaned_data.get('username')
            # email = form.cleaned_data.get('email')
            ######################### mail system ####################################
            # htmly = get_template('user/Email.html')
            # d = { 'username': username }
            # subject, from_email, to = 'welcome', 'your_email@gmail.com', email
            # html_content = htmly.render(d)
            # msg = EmailMultiAlternatives(subject, html_content, from_email, [to])
            # msg.attach_alternative(html_content, "text/html")
            # msg.send()
            ##################################################################
            # messages.success(request, f'Your account has been created ! You are now able to log in')
            return redirect('login')
        # else:
        #     render(request, 'login.html', {'form': form, "page_title":"Login",'errors':form.errors})

    else:
        form = UserRegisterForm()
    return render(request, 'signup.html', {'form': form, "page_title":"Sign Up",'errors':form.errors})


@login_required
def chatbot(request):
    return render(request,'chat.html',{"page_title" : "Chat"})

def about(request):
    return render(request,'about.html',{"page_title" : "About"})

################ login forms###################################################
# def login(request):
#     form = Login()
#     # message = ''
#     if request.method == 'POST':

#         form = Login(request.POST)
#         #AuthenticationForm_can_also_be_used__
  
#         username = request.POST['username']
#         password = request.POST['password']
#         user = authenticate(request, username = username, password = password)
#         if user is not None:
#             form = login(request, user)
#             messages.success(request, f' welcome {username} !!')
#             return redirect('index')
#         else:
#             messages.info(request, f'account done not exit plz sign in')
#     form = AuthenticationForm()
    # return render(request, 'login.html',{'form': form})

# def chat_reply_message(request):
#     from chatbot_application import chatbot
#     run_chatbot = chatbot.ChatBot()
#     msg = {'text' : run_chatbot.reply}
#     return render(request,msg)




