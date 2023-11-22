# Import all of the dependencies
import streamlit as st
import os
import time
import imageio 

import tensorflow as tf 
from utils import load_data, num_to_char
from modelutil import load_model

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Setup the sidebar with a dark background
with st.sidebar: 
    st.image('https://i.postimg.cc/MGkwqVkK/Speak-Easy.png', width=250)
    st.title('SpeakEasy')
    st.info('The objective of this project is to develop an end-to-end machine learning solution to detect words from a video of a person speaking.')
    
   # Scrolldown button in the sidebar
    if st.button('Meet the Team'):
        st.subheader('Team ID - 593015')
        st.write("1. Rudranil")
        st.write("2. Anurag")
        st.write("3. Karthik")
        st.write("4. Preetham")
# Set title and add some spacing
st.title('_SpeakEasy_ - :blue[Lip Reading using Deep Learning]') 
st.write("\n")

# Generating a list of options or videos 
options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Choose video', options)

# Generate two columns 
col1, col2 = st.columns(2)

if options: 

    # Rendering the video with a spinner
    with col1: 
        st.info('The video below displays the converted video in mp4 format')
        
        with st.spinner('Rendering video...'):
            file_path = os.path.join('..','data','s1', selected_video)
            os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')
            video = open('test_video.mp4', 'rb') 
            video_bytes = video.read() 
            st.video(video_bytes)

    # Add some spacing between the columns
    st.write("\n")

    # Model prediction with progress bar
    with col2: 
        st.info('This is all the machine learning model sees when making a prediction')
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        st.image('animation.gif', width=400) 

        st.info('This is the output of the machine learning model as tokens')

        # Simulate a time-consuming operation for model prediction
        with st.spinner('Performing Lip Reading...'):
            model = load_model()
            yhat = model.predict(tf.expand_dims(video, axis=0))
            decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
            st.text(decoder)
            
        # Convert prediction to text
        st.info('Decode the raw tokens into words')

        # Show and update progress bar
        bar = st.progress(50)
        time.sleep(1)
        bar.progress(100)

        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
        
# Additional elements
st.success('Speech Successfully Interpreted!')