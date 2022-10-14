import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.utils import img_to_array, load_img
from skimage.transform import resize
from skimage.io import imsave, imshow
from skimage.color import rgb2lab, lab2rgb

def main_colorization():
    
    st.header("Image Colorization")
    model = tf.keras.models.load_model("models/image-colorization.h5")

    image_file = st.file_uploader(
        "Upload image for testing", type=['jpeg', 'png', 'jpg', 'webp'])
   
    
    if st.button("Process"):
        image = Image.open(image_file)
        h,w = 256,256
                          
        img1_color=[]

        img1 = img_to_array(image)
        img1 = resize(img1 ,(256,256))
        img1_color.append(img1)
        
        img1_color = np.array(img1_color, dtype=float)
        img1_color = rgb2lab(1.0/255*img1_color)[:,:,:,0]
        img1_color = img1_color.reshape(img1_color.shape+(1,))
        output1 = model.predict(img1_color)
        output1 = output1*128

        result = np.zeros((256, 256, 3))
        result[:,:,0] = img1_color[0][:,:,0]
        result[:,:,1:] = output1[0]
        col1, col2 = st.columns([1,1])
        image = image.resize((h,w))
        
        with col1:
            st.text("Original Image")
            st.image(image)
        with col2 :
            st.text("Colourful Image")
            st.image(resize(lab2rgb(result),(256,256)))
        
    st.write("Image Colorization Notebook [link](https://github.com/bozkurtmert0/deep-learning-projects/blob/main/Image_Colorization.ipynb)")

if __name__ == '__main__':
    main_colorization()	   