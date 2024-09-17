import nltk
nltk.download('punkt')
nltk.download('stopwords')

import streamlit as st
import pickle 
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

f1=r"D:\Shinobu Python for Prac\vectorizer.pkl"
f2=r"D:\Shinobu Python for Prac\model.pkl"
with open(f1, 'rb') as file1:
    tk = pickle.load(file1)
with open(f2, 'rb') as file2:
    model = pickle.load(file2)


#tk = pickle.load(open(f1, 'rb'))
#model = pickle.load(open(f2, 'rb'))

st.title("AI Spam SMS and Email Detector")
st.write("A AI Model by Swastik/Shinobu Pythons")
    

input_sms = st.text_input("Enter your SMS or Email")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tk.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("This message is Spam")
    else:
        st.header("This message is Not Spam")
