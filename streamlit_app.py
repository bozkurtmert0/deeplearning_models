# importing libraries
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model = tf.keras.models.load_model("models/catsVSdogs.h5")

def main():
    st.title("Testing")

    st.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("**Select an option ** ")
    st.sidebar.write("")

    activities = [
                  "Cat vs Dog"]
    choice = st.sidebar.selectbox("select an option", activities)

    if choice == "Cat vs Dog":
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


if __name__ == "__main__":
    main()