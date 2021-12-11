import random
import json
import numpy as np
import nltk
import pickle
from nltk.stem import WordNetLemmatizer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('Chatbotmodel.h5')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if w == word:
                bag[i] = 1
                # if show_details:
                #     print('found in bag: %w' %word)
    return (np.array(bag))


def predict_class(sentence , model):
    bow = bag_of_words(sentence, words, show_details=False)
    res = model.predict(np.array([bow]))[0]

    ERROR_THRESHOLD= 0.25
    results= [[i,r] for i,r in enumerate(res) if r> ERROR_THRESHOLD]

    results.sort(key=lambda  x:x[1], reverse=True) # sort by highest probability
    return_list=[]
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list,intents_json):
    tag=intents_list[0]['intent']
    list_of_intents= intents_json['intents']
    for i in list_of_intents:
        if (i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(text):
    ints=predict_class(text,model)
    res=get_response(ints,intents)
    return res

print('presenting LDCE bot')

import tkinter

from tkinter import *


def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)
    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


base = Tk()
base.title("LDCE bot")
base.geometry("800x600")
base.resizable(width=FALSE, height=FALSE)


ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)
ChatLog.config(state=DISABLED)


scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set


SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#000000", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )


EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")

scrollbar.place(x=600,y=6, height=500)
ChatLog.place(x=6,y=6, height=500, width=600)
EntryBox.place(x=128, y=500, height=90, width=450)
SendButton.place(x=6, y=500, height=90)
base.mainloop()