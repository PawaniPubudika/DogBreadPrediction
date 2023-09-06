# Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model

# Loading the Model
model = load_model('/Users/pawanipubudika/Documents/ready/Dog bread prediction/src/dog_breed.h5')

# Name of Classes
CLASS_NAMES = ['Scottish Deerhound', 'Maltese Dog', 'Bernese Mountain Dog']

# Setting Title of App
st.title("Dog Breed Prediction")
st.markdown("<h3>Here we mainly predict three types of dogs</h3>",unsafe_allow_html=True)

# Create a column layout to display images and descriptions vertically
col1, col2, col3 = st.columns(3)

# Display the first image and description
with col1:
    st.image("/Users/pawanipubudika/Documents/ready/Dog bread prediction/testing data/AdobeStock_2808763-e1665449007319.jpeg", caption="Scottish Deerhound", use_column_width=True)

# Display the second image and description
with col2:
    st.image("/Users/pawanipubudika/Documents/ready/Dog bread prediction/testing data/licensed-image (1).jpeg", caption="Maltese Dog", use_column_width=True)

# Display the third image and description
with col3:
    st.image("/Users/pawanipubudika/Documents/ready/Dog bread prediction/testing data/licensed-image (2).jpeg", caption="Bernese Mountain Dog", use_column_width=True)

st.markdown("<h3>Upload an image of the dog</h3>",unsafe_allow_html=True)

# Uploading the dog image
dog_image = st.file_uploader("Choose an image...", type="jpg")
submit = st.button('Predict')

# On predict button click
if submit:
    if dog_image is not None:
        # Read the uploaded image using OpenCV
        opencv_image = cv2.imdecode(np.frombuffer(dog_image.read(), np.uint8), -1)
        
        if opencv_image is not None:
            # Display the image
            st.image(opencv_image, channels="BGR")
            
            # Resize the image
            opencv_image = cv2.resize(opencv_image, (224, 224))
            # Convert image to 4 Dimensions
            opencv_image = np.expand_dims(opencv_image, axis=0)
            
            # Make Prediction
            Y_pred = model.predict(opencv_image)
            
            st.title(str("The Dog Breed is " + CLASS_NAMES[np.argmax(Y_pred)]))
        else:
            st.warning("Error: Unable to decode the uploaded image.")
