import streamlit as st
import pickle
from huggingface_hub import hf_hub_download
from fastai.vision.all import *
from PIL import Image

# Your Hugging Face repo details
HF_REPO_ID = "YOUR_USERNAME/animal-classifier"
HF_MODEL_FILE = "nature_model.pkl"
HF_CLASSES_FILE = "classes.pkl"

@st.cache_resource
def load_assets():
    # Download model file from Hugging Face Hub
    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_MODEL_FILE
    )
    # Download classes file
    classes_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_CLASSES_FILE
    )
    # Load them
    learn = load_learner(model_path)
    with open(classes_path, 'rb') as f:
        classes = pickle.load(f)
    return learn, classes

learn, classes = load_assets()

st.title("🐾 Animal Classifier")
st.write("Upload a photo of an animal. The model predicts its biological class.")
st.write(f"**Classes:** {', '.join(classes)}")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)

    with st.spinner('Classifying...'):
        pred_class, pred_idx, probs = learn.predict(PILImage.create(image))

    confidence = float(probs[pred_idx]) * 100
    st.success(f"**Prediction: {pred_class}** ({confidence:.1f}% confidence)")

    st.subheader("All Probabilities")
    prob_dict = {classes[i]: float(probs[i]) for i in range(len(classes))}
    st.bar_chart(prob_dict)