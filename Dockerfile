FROM python:3.8

WORKDIR /deeplearning_models_heroku
ADD models /deeplearning_models_heroku/models
ADD streamlit_app.py .

RUN pip install numpy tensorflow-cpu opencv-python-headless streamlit Pillow scikit-learn pandas pickleshare keras requests flask

CMD ["streamlit", "run" ,"./streamlit_app.py"] 