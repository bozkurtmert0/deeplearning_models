import streamlit as st
import os
import pandas as pd
from face_detec import main_face 
from movie_rec import main_movie
from disaster_twet import main_twet
from catvsdog import main_catvsdog

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# -------------------------------------------------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide")
    st.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.subheader("Select an option")
    activities = [
        "Cats vs Dogs", "Disaster Tweet Classification", "Movie Recommender", "Face Detection"]
    choice = st.sidebar.selectbox("", activities)

# ------------Cats Vs Dogs ----------------------------------------------------------------

    if choice == "Cats vs Dogs":
        main_catvsdog()
# ------------------------------------------------------------------------
    if choice == "Disaster Tweet Classification":
        main_twet()

# ----------------------------------------------------------------------------------------------------------------
    if choice == "Movie Recommender":
        main_movie()
# -------------------------------------------------------------------------------
    if choice == "Face Detection": 
        main_face()

if __name__ == '__main__':
    main()
