import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import pandas as pd
import math

# Load model
model = tf.keras.models.load_model('model/model2.h5')

# Preprocess the image
def preprocess_image(image, target_size):
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    # Resize the image
    image = image.resize(target_size)
    
    # Convert image to grayscale, but retain 3 channels
    image_gray = image.convert('L')  # Convert to grayscale (1 channel)
    image_gray = np.array(image_gray, dtype=np.float32)  # Convert to numpy array
    
    # Stack the grayscale image to 3 channels (R, G, B)
    image_gray = np.stack([image_gray] * 3, axis=-1)  # Shape becomes (height, width, 3)
    
    # Normalize using preprocess_input (MobileNet expects RGB image)
    image_array = preprocess_input(image_gray)
    
    # Add batch dimension (1, height, width, channels)
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

# Classify the image
def classify_image(image, model):
    processed_image = preprocess_image(image, target_size=(64, 64))
    predictions = model.predict(processed_image)
    return predictions

# Mapping for the class labels
label = {0: 'ba', 1: 'ca', 2: 'da', 3: 'ga', 4: 'ha', 5: 'ja', 6: 'ka', 7: 'la', 8: 'ma', 9: 'na', 10: 'nga', 11: 'nya', 12: 'pa', 13: 'ra', 14: 'sa', 15: 'ta', 16: 'wa', 17: 'ya'}

# UI
st.title("Balinese Aksara Classifier")
st.write("Upload one or more images to classify.")

# File uploader for multiple images
uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files is not None:
    # Prepare lists to hold data for the table
    image_names = []
    predicted_classes = []
    prediction_probs = []
    prob = []
    result = []
    
    # Iterate over each uploaded image
    for uploaded_file in uploaded_files:
        # Load and preprocess the image
        image = Image.open(uploaded_file)
        predictions = classify_image(image, model)
        class_idx = np.argmax(predictions[0])
        
        
        # Store data
        image_names.append(uploaded_file.name)
        predicted_classes.append(f"{label[class_idx]} ({class_idx})")
        prediction_probs.append(predictions[0])
        prob.append(np.max(predictions[0]))

        result.append(
            {
            'Image': uploaded_file.name,
            'Predicted Class': f"{class_idx} ({label[class_idx]})",
            'Prob' : np.max(predictions[0]),
            'Prediction Probabilities': predictions[0]
            }
        )
    
    # Create a DataFrame to display in a table
    results = {
        'Image': image_names,
        'Predicted Class': predicted_classes,
        'Prob' : prob,
        'Prediction Probabilities': prediction_probs
    }

    # Display the table with results
    st.table(results)

    # # Create a DataFrame from the data
    # df = pd.DataFrame(result, columns=["Image", "Predicted Class", "Prob", "Prediction Probabilities"])

    # # Show as a static table
    # st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)

    st.write('### Table of images')

    #calculate media rows and columns
    image_count = len(uploaded_files)
    rows = math.ceil(image_count/5)
    columns = 5
    

    for i in range(rows): # number of rows in your table! = 2
        cols = st.columns(columns) # number of columns in each row! = 2
        # first column of the ith row
        for j in range(columns):
        # Calculate the index of the current image
            index = i * columns + j
            if index < image_count:  # Ensure that index is within bounds
                # Display the image in the current column
                cols[j].image(uploaded_files[index], width=100, caption=uploaded_files[index].name)