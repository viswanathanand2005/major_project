import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input
import cv2
from tensorflow.keras.preprocessing import image
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

# Custom CSS for background and text colors

st.markdown(
    """
    <style>
.stApp {
    background-color: #1A2A4F !important;
    color: #ffffff !important;
}

html, body, [class*="css"] {
    color: #ffffff !important;
}

hr {
    border: 1px solid white !important;
}

/* Hiding the header/footer */
header[data-testid="stHeader"] { display: none; }
footer { display: none; }

/* Input field styling */
div[data-testid="stTextArea"] textarea,
textarea[role="textbox"] {
    color: #ffffff;
}

/* Checkbox styling */
div[data-testid="stCheckbox"] * {
    color: #ffffff;
}

/* Label styling */
label, .css-1aumxhk, .stTextInput label {
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True
)


# We only saved the MODEL WEIGHTS, we need to REDEFINE THE MODEL ARCHITECTURE with all the preprocessing as well.

classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy',
           'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy' , 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
           'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
           'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)',
           'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
           'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
           'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
           'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
           'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

st.title("Plant Disease Detection App")
st.divider()
st.write("Upload a leaf image to identify the disease")

upload_file = st.file_uploader("Upload here", type = ['png', 'jpg', 'jpeg'])

@st.cache_resource
def load_model():

    inputs = Input(shape=(224, 224, 3))
    
    resnet_base = ResNet50(weights='imagenet', include_top=False, input_tensor=(inputs))
    
    resnet_base.trainable = False
    
    x = resnet_base.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation='relu', name="head_dense")(x)
    x = layers.Dropout(0.4, name="head_dropout")(x)
    outputs = layers.Dense(len(classes), activation='softmax', name="predictions")(x)

    model = models.Model(inputs, outputs)

    weights = np.load("resnet_weights.npz", allow_pickle=True)
    model.set_weights([weights[key] for key in weights])

    return model

model = load_model()

if upload_file is not None:
    image = Image.open(upload_file).convert("RGB")
    
    # preprocessing the image
    image = image.resize((224,224))
    image_array = np.array(image)
    image_batch = np.expand_dims(image_array, axis=0)
    image_batch = preprocess_input(image_batch)
    
    predictions = model.predict(image_batch)
    confidence = np.max(predictions)
    predicted_class = classes[np.argmax(predictions)]
    
    # Defining readable mapping for easier understanding
    class_descriptions = {
        'Apple___Apple_scab': "Apple plant affected by Apple scab",
        'Apple___Black_rot': "Apple plant affected by Black rot",
        'Apple___Cedar_apple_rust': "Apple plant affected by Cedar apple rust",
        'Apple___healthy': "Apple plant — healthy and disease-free",
        'Blueberry___healthy': "Blueberry plant — healthy and disease-free",
        'Cherry_(including_sour)___Powdery_mildew': "Cherry plant affected by Powdery mildew",
        'Cherry_(including_sour)___healthy': "Cherry plant — healthy and disease-free",
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "Corn plant affected by Gray leaf spot",
        'Corn_(maize)___Common_rust_': "Corn plant affected by Common rust",
        'Corn_(maize)___Northern_Leaf_Blight': "Corn plant affected by Northern Leaf Blight",
        'Corn_(maize)___healthy': "Corn plant — healthy and disease-free",
        'Grape___Black_rot': "Grape plant affected by Black rot",
        'Grape___Esca_(Black_Measles)': "Grape plant affected by Esca (Black Measles)",
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': "Grape plant affected by Leaf blight (Isariopsis Leaf Spot)",
        'Grape___healthy': "Grape plant — healthy and disease-free",
        'Orange___Haunglongbing_(Citrus_greening)': "Orange plant affected by Huanglongbing (Citrus greening)",
        'Peach___Bacterial_spot': "Peach plant affected by Bacterial spot",
        'Peach___healthy': "Peach plant — healthy and disease-free",
        'Pepper,_bell___Bacterial_spot': "Bell pepper plant affected by Bacterial spot",
        'Pepper,_bell___healthy': "Bell pepper plant — healthy and disease-free",
        'Potato___Early_blight': "Potato plant affected by Early blight",
        'Potato___Late_blight': "Potato plant affected by Late blight",
        'Potato___healthy': "Potato plant — healthy and disease-free",
        'Raspberry___healthy': "Raspberry plant — healthy and disease-free",
        'Soybean___healthy': "Soybean plant — healthy and disease-free",
        'Squash___Powdery_mildew': "Squash plant affected by Powdery mildew",
        'Strawberry___Leaf_scorch': "Strawberry plant affected by Leaf scorch",
        'Strawberry___healthy': "Strawberry plant — healthy and disease-free",
        'Tomato___Bacterial_spot': "Tomato plant affected by Bacterial spot",
        'Tomato___Early_blight': "Tomato plant affected by Early blight",
        'Tomato___Late_blight': "Tomato plant affected by Late blight",
        'Tomato___Leaf_Mold': "Tomato plant affected by Leaf Mold",
        'Tomato___Septoria_leaf_spot': "Tomato plant affected by Septoria leaf spot",
        'Tomato___Spider_mites Two-spotted_spider_mite': "Tomato plant affected by Two-spotted spider mite",
        'Tomato___Target_Spot': "Tomato plant affected by Target Spot",
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "Tomato plant affected by Tomato Yellow Leaf Curl Virus",
        'Tomato___Tomato_mosaic_virus': "Tomato plant affected by Tomato mosaic virus",
        'Tomato___healthy': "Tomato plant — healthy and disease-free"
    }
    
    readable_prediction = class_descriptions.get(predicted_class, predicted_class)

    st.subheader(f"Prediction: {readable_prediction}")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")

    def make_gradcam_heatmap(img_array, model, last_conv_layer_name='conv5_block3_out'):
        last_conv_layer = model.get_layer(last_conv_layer_name)
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [last_conv_layer.output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
        return heatmap

    heatmap = make_gradcam_heatmap(image_batch, model)

    # overlay and plot
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    heatmap_resized = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)
    
    superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    st.subheader("Grad-CAM Visualization")
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", width=350)
    with col2:
        st.image(superimposed_img_rgb, caption="Grad-CAM Overlay", width=350)
        
    load_dotenv()
    
    
    HF_TOKEN = st.secrets.get("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

    client = InferenceClient(
        provider="featherless-ai",
        api_key=HF_TOKEN,
    )
    
    if "healthy" in readable_prediction.lower():
        condition_type = "healthy"
    else:
        condition_type = "diseased"
        
    prompt = f"""
    You are an expert plant pathologist. 
    The model detected that the plant is **{readable_prediction}**, which means it is {condition_type}.
    Give concise points of actionable care advice.
    Avoid saying phrases like 'if healthy' or 'if diseased' — speak directly for this condition only.
    Each point should be very detailed and long.
    """

    completion = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        messages=[
            {"role": "system", "content": "You are a concise, professional plant health expert. Respond with direct care instructions only. Avoid filler phrases like 'Of course' or 'Here’s your advice.'"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
    )

    response = completion.choices[0].message["content"]

    
    st.divider()
    st.subheader("Plant Health Advice:")
    st.write(response)