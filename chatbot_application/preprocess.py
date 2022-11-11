# import re

# lines = open('../input/chatbot-data/cornell movie-dialogs corpus/movie_lines.txt', encoding='utf-8',
#              errors='ignore').read().split('\n')

# convers = open('../input/chatbot-data/cornell movie-dialogs corpus/movie_conversations.txt', encoding='utf-8',
#              errors='ignore').read().split('\n')


import re

lines = open('D:\\python files\\CHATBOTS\\new\\cornell_movie_dialogs_corpus\\cornell movie-dialogs corpus\\movie_lines.txt', encoding='utf-8',
             errors='ignore').read().split('\n')

convers = open('D:\\python files\\CHATBOTS\\new\\cornell_movie_dialogs_corpus\\cornell movie-dialogs corpus\\movie_conversations.txt', encoding='utf-8',
             errors='ignore').read().split('\n')


print(len(lines))
















# DATA PREPROCESSING

exchn = []
for conver in convers:
    exchn.append(conver.split(' +++$+++ ')[-1][1:-1].replace("'", " ").replace(",","").split())

diag = {}
for line in lines:
    diag[line.split(' +++$+++ ')[0]] = line.split(' +++$+++ ')[-1]



## delete
del(lines, convers, conver, line)



questions = []
answers = []

for conver in exchn:
    for i in range(len(conver) - 1):
        questions.append(diag[conver[i]])
        answers.append(diag[conver[i+1]])
        
        
        

## delete
del(diag, exchn, conver, i)


###############################
#        max_len = 13         #
###############################

sorted_ques = []
sorted_ans = []
for i in range(len(questions)):
    if len(questions[i]) < 13:
        sorted_ques.append(questions[i])
        sorted_ans.append(answers[i])

###############################
#                             #
###############################

def clean_text(txt):
    txt = txt.lower()
    txt = re.sub(r"i'm", "i am", txt)
    txt = re.sub(r"he's", "he is", txt)
    txt = re.sub(r"she's", "she is", txt)
    txt = re.sub(r"that's", "that is", txt)
    txt = re.sub(r"what's", "what is", txt)
    txt = re.sub(r"where's", "where is", txt)
    txt = re.sub(r"\'ll", " will", txt)
    txt = re.sub(r"\'ve", " have", txt)
    txt = re.sub(r"\'re", " are", txt)
    txt = re.sub(r"\'d", " would", txt)
    txt = re.sub(r"won't", "will not", txt)
    txt = re.sub(r"can't", "can not", txt)
    txt = re.sub(r"[^\w\s]", "", txt)
    return txt

clean_ques = []
clean_ans = []

for line in sorted_ques:
    clean_ques.append(clean_text(line))
        
for line in sorted_ans:
    clean_ans.append(clean_text(line))

## delete
del(answers, questions, line)

###############################
#                             #
###############################

for i in range(len(clean_ans)):
    clean_ans[i] = ' '.join(clean_ans[i].split()[:11])

###############################
#                             #
###############################

del(sorted_ans, sorted_ques)


## trimming
clean_ans=clean_ans[:30000]
clean_ques=clean_ques[:30000]
## delete


###  count occurences ###
word2count = {}

for line in clean_ques:
    for word in line.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
for line in clean_ans:
    for word in line.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

## delete
del(word, line)

###  remove less frequent ###
thresh = 5

vocab = {}
word_num = 0
for word, count in word2count.items():
    if count >= thresh:
        vocab[word] = word_num
        word_num += 1
        
## delete
del(word2count, word, count, thresh)       
del(word_num)        



for i in range(len(clean_ans)):
    clean_ans[i] = '<SOS> ' + clean_ans[i] + ' <EOS>'

tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
x = len(vocab)
for token in tokens:
    vocab[token] = x
    x += 1
  

vocab['cameron'] = vocab['<PAD>']
vocab['<PAD>'] = 0

## delete
del(token, tokens) 
del(x)

### inv answers dict ###
inv_vocab = {w:v for v, w in vocab.items()}



## delete
del(i)

encoder_inp = []
for line in clean_ques:
    lst = []
    for word in line.split():
        if word not in vocab:
            lst.append(vocab['<OUT>'])
        else:
            lst.append(vocab[word])
        
    encoder_inp.append(lst)

decoder_inp = []
for line in clean_ans:
    lst = []
    for word in line.split():
        if word not in vocab:
            lst.append(vocab['<OUT>'])
        else:
            lst.append(vocab[word])        
    decoder_inp.append(lst)

### delete
del(clean_ans, clean_ques, line, lst, word)

import tensorflow
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import tensorflow.keras.utils.pad_sequences as pad_sequences


encoder_inp = pad_sequences(encoder_inp, 13, padding='post', truncating='post')
decoder_inp = pad_sequences(decoder_inp, 13, padding='post', truncating='post')


decoder_final_output = []
for i in decoder_inp:
    decoder_final_output.append(i[1:]) 

decoder_final_output = pad_sequences(decoder_final_output, 13, padding='post', truncating='post')


del(i)





















































# decoder_final_output, decoder_final_input, encoder_final, vocab, inv_vocab

VOCAB_SIZE = len(vocab)
MAX_LEN = 13

print(decoder_final_output.shape, decoder_inp.shape, encoder_inp.shape, len(vocab), len(inv_vocab), inv_vocab[0])




print(inv_vocab[16])










#print(len(decoder_final_input), MAX_LEN, VOCAB_SIZE)
#decoder_final_input[0]
#decoder_output_data = np.zeros((len(decoder_final_input), MAX_LEN, VOCAB_SIZE), dtype="float32")
#print(decoder_output_data.shape)
#decoder_final_input[80]
from tensorflow.keras.utils import to_categorical
decoder_final_output = to_categorical(decoder_final_output, len(vocab))
print(decoder_final_output.shape)








# Glove Embedding
import numpy as np
embeddings_index = {}
with open('D:\\python files\\CHATBOTS\\new\\glove.6B.50d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

print("Glove Loded!")










embedding_dimention = 50
def embedding_matrix_creater(embedding_dimention, word_index):
    embedding_matrix = np.zeros((len(word_index)+1, embedding_dimention))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
          # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
embedding_matrix = embedding_matrix_creater(50, word_index=vocab)    
del(embeddings_index)
print(embedding_matrix.shape)



print(embedding_matrix[0])

# print(embedding_matrix[0])




from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input, Bidirectional, Concatenate, Dropout, Attention




embed = Embedding(VOCAB_SIZE+1,50,input_length=13,trainable=True)

embed.build((None,))
embed.set_weights([embedding_matrix])


enc_inp = Input(shape=(13, ))
















#embed = Embedding(VOCAB_SIZE+1, 50, mask_zero=True, input_length=13)(enc_inp)
enc_embed = embed(enc_inp)
enc_lstm = Bidirectional(LSTM(400, return_state=True, dropout=0.05, return_sequences = True))

encoder_outputs, forward_h, forward_c, backward_h, backward_c = enc_lstm(enc_embed)

state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])

enc_states = [state_h, state_c]


dec_inp = Input(shape=(13, ))
dec_embed = embed(dec_inp)
dec_lstm = LSTM(400*2, return_state=True, return_sequences=True, dropout=0.05)
output, _, _ = dec_lstm(dec_embed, initial_state=enc_states)

# attention
from AttentionLayer import AttentionLayer
attn_layer = AttentionLayer()
attn_op, attn_state = attn_layer([encoder_outputs, output])
decoder_concat_input = Concatenate(axis=-1)([output, attn_op])


dec_dense = Dense(VOCAB_SIZE, activation='softmax')
final_output = dec_dense(decoder_concat_input)

model = Model([enc_inp, dec_inp], final_output)

model.summary()












import tensorflow.keras
import tensorflow as tf




model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])



model.fit([encoder_inp, decoder_inp], decoder_final_output, epochs=25, batch_size=24, validation_split=0.15)


# model.fit([encoder_inp, decoder_inp], decoder_final_output, epochs=10, batch_size=24, validation_split=0.15)


model.save('chatbot.h5')
model.save_weights('chatbot_weights.h5')


