import pandas as pd
import streamlit as st 
import pickle
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer

def main_twet():

    st.header("Disaster Tweet Classification")
    filename = 'models/finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    data1 = pd.read_csv("models/data1.csv")

    train_x, test_x, train_y, test_y = model_selection.train_test_split(data1["text"],
                                                                            data1["target"], random_state=42)

    sentences = ["Just happened a terrible car crash",
                     "We're shaking...It's an earthquake",
                     "there is a forest fire at spot pond, geese are fleeing across the street, I cannot save them all",
                     "Paradise ,the bitches say im hot i say no bitch im blazing",
                     "Refugio oil spill may have been costlier bigger than projected",
                     "someone hold my hand and tell me ITS OKAY because I am having a panic attack for no reason"
                     ]

    option = st.selectbox(
            'You can select here', sentences)

    if st.button("Process from select box"):
        option = pd.Series(option)
        vectorizer = CountVectorizer()
        vectorizer.fit(train_x)
        option = vectorizer.transform(option)
        result = loaded_model.predict(option)
        if result == 1:
            st.header("Disaster")
        if result == 0:
            st.header("Not Disaster")

    input = st.text_input("Custom text")

    if st.button("Process custom text"):
        input = pd.Series(input)
        vectorizer = CountVectorizer()
        vectorizer.fit(train_x)
        input = vectorizer.transform(input)
        result = loaded_model.predict(input)

        if result == 1:
            st.header("Disaster")

        if result == 0:
            st.header("Not Disaster")
            

if __name__ == '__main__':
    main_twet()	