# importing libraries
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os
import tensorflow as tf
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn import model_selection
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def main():
    st.title("Testing")

    st.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("**Select an option ** ")
    st.sidebar.write("")

    activities = [
                  "Cat vs Dog","Disaster Tweet Classification"]
    choice = st.sidebar.selectbox("select an option", activities)

#------------Cats Vs Dog ----------------------------------------------------------------
    
    if choice == "Cat vs Dog":
        
        model = tf.keras.models.load_model("models/catsVSdogs.h5")
        
        image_file = st.file_uploader(
            "Upload image for testing", type=['jpeg', 'png', 'jpg', 'webp'])
 
        if st.button("Process"):
            image = Image.open(image_file)       
            #image = cv2.imread (image_file)
            image = np.array(image.convert('RGB'))
            image = cv2.resize(image,(224, 224))
            image = np.reshape(image,[1,224, 224,3])      
            
            FRAME_WINDOW = st.image([])
            
            classes = model.predict(image)
            print(classes)
            if classes > 0.5:
                st.header("Dog")
                st.subheader(classes)
            if classes < 0.5 :
                st.header("Cat")
                st.subheader(1-classes)
           
            image1 = Image.open(image_file)
            FRAME_WINDOW.image(image1)
#------------------------------------------------------------------------
    if choice == "Disaster Tweet Classification":
         
        filename = 'models/finalized_model.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        data1 = pd.read_csv("models/data1.csv")
        
        train_x, test_x, train_y, test_y = model_selection.train_test_split(data1["text"],
                                                                     data1["target"],random_state =42)
        
        
        
        sentences = ["Just happened a terrible car crash",
                     "We're shaking...It's an earthquake",
                     "there is a forest fire at spot pond, geese are fleeing across the street, I cannot save them all",
                     "Paradise ,the bitches say im hot i say no bitch im blazing",
                     "Refugio oil spill may have been costlier bigger than projected",
                     "someone hold my hand and tell me ITS OKAY because I am having a panic attack for no reason"
                     ]
        
        option = st.selectbox(
        'You can select here',sentences)
        
        #pd.Series(option)
        #train_x = train_x.values.astype('U')
        if st.button("Process from select box"):
            #pd.Series(option)
            option = pd.Series(option)
            vectorizer = CountVectorizer()
            vectorizer.fit(train_x)
            option = vectorizer.transform(option)
            result = loaded_model.predict(option)

            if result == 1 :
                st.header("Disaster")
            
            if result == 0 :
                st.header("Not Disaster")
                
        input = st.text_input("Custom text")
        
        if st.button("Process custom text"):
            input = pd.Series(input)
            vectorizer = CountVectorizer()
            vectorizer.fit(train_x)
            input = vectorizer.transform(input)
            result = loaded_model.predict(input)

            if result == 1 :
                st.header("Disaster")
            
            if result == 0 :
                st.header("Not Disaster")

if __name__ == "__main__":
    main()
