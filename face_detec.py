import streamlit as st 
import cv2
from PIL import Image,ImageEnhance
import numpy as np 
import os

face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('models/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('models/haarcascade_smile.xml')

@st.cache
def load_image(img):
	im = Image.open(img)
	return im

def detect_faces(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img,1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return img,faces 

def main_face():
	
	st.title("Face Detection App")
	image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])

	if image_file is not None:
		our_image = Image.open(image_file)
		st.text("Original Image")
		# st.write(type(our_image))
		st.image(our_image,width=300)
		
		if st.button("Process"):
			result_img,result_faces = detect_faces(our_image)
			st.image(result_img)

			st.success("Found {} faces".format(len(result_faces)))
			#elif feature_choice == 'Smiles':
			#	result_img = detect_smiles(our_image)
			#	st.image(result_img)

if __name__ == '__main__':
	main_face()	
