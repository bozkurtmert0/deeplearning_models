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
import requests
import zipfile
import bz2
import pickle
import _pickle as cPickle
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


#------------------For Docker and Heroku ---------------------------------------
from flask import Flask
import os
app = Flask(__name__)
port = int(os.environ.get("PORT", 5000))

#-----------------------------------------------------------------------------------

movies = pickle.load(open('models/movie_list.pkl','rb'))
<<<<<<< HEAD
=======
#archive = zipfile.ZipFile('models/similarity.zip', 'r')
#similarity = archive.read("similarity.pkl")
>>>>>>> 566b44b4ef7a70b9785d1003451a155432e16eea
def decompress_pickle(file):
  data = bz2.BZ2File(file, "rb")
  data = cPickle.load(data)
  return data

similarity = decompress_pickle('models/similarity.pbz2') 


def fetch_poster(movie_id):
    
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        # fetch the movie poster
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)

    return recommended_movie_names,recommended_movie_posters

#-------------------------------------------------------------------------------------------------------------
def main():
    st.title("Testing")

    st.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("**Select an option ** ")
    st.sidebar.write("")

    activities = [
                  "Cat vs Dog","Disaster Tweet Classification", "Movie Recommender"]
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
                
#----------------------------------------------------------------------------------------------------------------

    if choice == "Movie Recommender":
        st.header('Movie Recommender System')
        
        
        
        #similarity = pickle.load(open('models/similarity.pkl','rb'))

        movie_list = movies['title'].values
        selected_movie = st.selectbox(
        "Type or select a movie from the dropdown",
        movie_list
                )

        if st.button('Show Recommendation'):
            recommended_movie_names,recommended_movie_posters = recommend(selected_movie)
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.text(recommended_movie_names[0])
                st.image(recommended_movie_posters[0])
            with col2:
                st.text(recommended_movie_names[1])
                st.image(recommended_movie_posters[1])

            with col3:
                st.text(recommended_movie_names[2])
                st.image(recommended_movie_posters[2])
            with col4:
                st.text(recommended_movie_names[3])
                st.image(recommended_movie_posters[3])
            with col5:
                st.text(recommended_movie_names[4])
                st.image(recommended_movie_posters[4])

<<<<<<< HEAD

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=port)
=======
if __name__ == "__main__":
    main()
>>>>>>> 566b44b4ef7a70b9785d1003451a155432e16eea
