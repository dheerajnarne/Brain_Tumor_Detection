from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the pre-trained model
model = load_model("E:\Brain_Tumor_Detection\Brain_Tumor_Detection.h5")

# Class labels
CLASS_LABELS = ['The person is suffering with Glioma', 'THe person is suffering with Meningioma', 'The person has No Tumor']

# Function to preprocess and predict image class
def predict_image_class(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match model input shape
    img_array = img_array / 255.0  # Normalize the image
    
    # Make prediction
    prediction = model.predict(img_array)
    
    # Get predicted class label
    predicted_class_index = np.argmax(prediction)
    
    # Map predicted class index to actual class label
    predicted_class_label = CLASS_LABELS[predicted_class_index]
    
    # Return predicted class label
    return predicted_class_label

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get uploaded image file
        image_file = request.files['image']
        
        # Save the image file temporarily
        image_path = 'temp_image.jpg'
        image_file.save(image_path)
        
        # Predict class label
        predicted_class = predict_image_class(image_path)
        
        # Remove the temporary image file
        os.remove(image_path)
        
        # Construct styled HTML response
        styled_response = f"""
        <div style=" margin-top: 20%; background-color: #f7fafc; border: 1px solid #e2e8f0; border-radius: 0.375rem; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); transition: box-shadow 0.2s ease-in-out;">
            <h2 style="font-size: 2.25rem; padding-left:32%;color: #4a5568;">{predicted_class}</h2>
        </div>
        """
        
        # Return styled HTML response
        return styled_response

if __name__ == '__main__':
    app.run(debug=True)
