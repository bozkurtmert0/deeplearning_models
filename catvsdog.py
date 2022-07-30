import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

def main_catvsdog():
    
    st.header("Cats Vs Dogs")
    model = tf.keras.models.load_model("models/catsVSdogs.h5")

    image_file = st.file_uploader(
        "Upload image for testing", type=['jpeg', 'png', 'jpg', 'webp'])

    if st.button("Process"):
        image = Image.open(image_file)
        #image = cv2.imread (image_file)
        image = np.array(image.convert('RGB'))
        image = cv2.resize(image, (224, 224))
        image = np.reshape(image, [1, 224, 224, 3])

        FRAME_WINDOW = st.image([])

        classes = model.predict(image)
        if classes > 0.5:
            st.header("Dog")
            st.subheader(classes)
        if classes < 0.5:
            st.header("Cat")
            st.subheader(1-classes)

        image1 = Image.open(image_file)
        FRAME_WINDOW.image(image1)
        

if __name__ == '__main__':
    main_catvsdog()	   