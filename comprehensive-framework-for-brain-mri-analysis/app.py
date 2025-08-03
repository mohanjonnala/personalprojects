# app.py

from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import io
import base64
import cv2
from keras import Sequential
from keras.layers import Dense
import tensorflow_hub as hub
import nibabel as nib
from keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'tif', 'tiff'}
smooth = 100
H = 256
W = 256

# Load pre-trained models for brain tumor classification
path3 = "https://tfhub.dev/google/efficientnet/b0/classification/"

efficient_model = hub.KerasLayer(path3, input_shape=(224, 224, 3), trainable=False)
num_class_brain = 4

efficient_pre_model_brain = Sequential()
efficient_pre_model_brain.add(efficient_model)
efficient_pre_model_brain.add(Dense(units=num_class_brain, activation="softmax"))
efficient_pre_model_brain.load_weights('models/brain_model.h5')

class_labels_brain = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

def preprocess_image_brain(image):
    image = cv2.resize(image, (224, 224))
    image = np.array(image)
    image = image / 255.0
    return image

# Load pre-trained model for image segmentation
# Assuming you have a function for image segmentation, replace the following line with your code
def dice_coef(y_true, y_pred):
    y_truef = K.flatten(y_true)
    y_predf = K.flatten(y_pred)
    And = K.sum(y_truef * y_predf)
    return (2 * And + smooth) / (K.sum(y_truef) + K.sum(y_predf) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

model = load_model('models/unet_brain_mri_seg.hdf5', custom_objects={'dice_coef_loss': dice_coef_loss, 'iou': iou, 'dice_coef': dice_coef})

def read_image(image):
    image = cv2.resize(image, (W, H))
    x = image / 255.0
    x = np.expand_dims(x, axis=0)
    y_pred = model.predict(x, verbose=0)[0]
    y_pred = np.squeeze(y_pred, axis=-1)
    y_pred = y_pred >= 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred * 255
    return y_pred

# Load pre-trained models for survival prediction
# ... (existing code for survival prediction)
model_survival = load_model('models/surv_pred3.h5', compile=False)

def preprocess_image_survival(image_path, age):
    img = nib.load(image_path)
    input_image_array = img.get_fdata()
    input_image_array = (input_image_array - np.mean(input_image_array)) / np.std(input_image_array)

    depth_slice = input_image_array[:, :, 75]
    input_image_array = np.stack([depth_slice] * 5, axis=-1)
    input_image_array = np.expand_dims(input_image_array, axis=0)

    return input_image_array, age

def predict_survival(input_image, age):
    predictions = model_survival.predict([input_image, np.array([[age]])])
    predicted_class = np.argmax(predictions)
    return predicted_class, predictions.flatten()


@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_class_brain = None
    efficient_pred_class = None
    predicted_class_survival = None
    probabilities_survival = None
    survival_days = None
    input_image_base64 = None
    mask_image_base64 = None

    input_image = None  # Initialize input_image here
    mask_image = None   # Initialize mask_image here

    if request.method == 'POST':
        if 'file' in request.files:
            # Process brain tumor classification
            file_brain = request.files['file']
            img_brain = Image.open(file_brain)
            img_brain = np.array(img_brain)
            img_brain = preprocess_image_brain(img_brain)

            efficient_pred_brain = efficient_pre_model_brain.predict(np.expand_dims(img_brain, axis=0))
            efficient_pred_class = class_labels_brain[np.argmax(efficient_pred_brain)]

        if 'file_survival' in request.files:
            # Process survival prediction
            file_survival = request.files['file_survival']
            age_survival = int(request.form['age_survival'])

            image_path_survival = f"uploads/{file_survival.filename}"
            file_survival.save(image_path_survival)

            input_image_survival, age_survival = preprocess_image_survival(image_path_survival, age_survival)
            predicted_class_survival, probabilities_survival = predict_survival(input_image_survival, age_survival)
            if predicted_class_survival == 0:
                survival_days = "Less than 300 days"
            elif predicted_class_survival == 1:
                survival_days = "Between 300 and 450 days"
            else:
                survival_days = "More than 450 days"

        if 'image' in request.files:
            # Get the uploaded file from the form
            uploaded_file = request.files['image']

            # Check if a file was actually uploaded
            if uploaded_file.filename != '':
                # Read the image from the uploaded file
                image = Image.open(uploaded_file)
                image_np = np.array(image)

                # Process the image
                y_pred = read_image(image_np)

                # Convert images to base64 for rendering in HTML template
                input_buffer = io.BytesIO()
                output_buffer = io.BytesIO()
                input_pil_image = Image.fromarray(image_np)
                output_pil_image = Image.fromarray(y_pred)
                input_pil_image.save(input_buffer, format='JPEG')
                output_pil_image.save(output_buffer, format='JPEG')
                input_image = base64.b64encode(input_buffer.getvalue()).decode('utf-8')
                mask_image = base64.b64encode(output_buffer.getvalue()).decode('utf-8')

    return render_template('index.html', efficient_pred=efficient_pred_class,
                           predicted_class_survival=predicted_class_survival,
                           probabilities_survival=probabilities_survival,
                           survival_days=survival_days,
                           input_image=input_image,
                           mask_image=mask_image)



# def image_to_base64(image):
#     img_byte_array = io.BytesIO()
#     image.save(img_byte_array, format='JPEG')
#     img_byte_array = img_byte_array.getvalue()
#     img_base64 = 'data:image/jpeg;base64,' + str(base64.b64encode(img_byte_array), 'utf-8')
#     return img_base64


if __name__ == '__main__':
    app.run(debug=True)
