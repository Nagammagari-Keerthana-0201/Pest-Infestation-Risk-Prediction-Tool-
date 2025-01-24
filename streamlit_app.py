import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import joblib
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from pest_image_processing import categories

st.markdown(
    """
    <style>
    body { background-color:#f0faff; color: #333333; }
    .css-1aumxhk { background-color: #ffffff; border: 1px solid #dddddd; border-radius: 8px; }
    .stTitle { color: #4CAF50; }
    button { background-color: #4CAF50; color: white; border: none; padding: 10px 20px; font-size: 16px; border-radius: 5px; cursor: pointer; }
    button:hover { background-color: #45a049; }
    </style>
    """,
    unsafe_allow_html=True
)
pest_recommendations = {
    "ants": "Use ant baits or natural repellents like lemon juice or vinegar.",
    "bees": "Avoid harmful pesticides; use smoke to relocate bees.",
    "beetle": "Introduce natural predators or use neem oil.",
    "catterpillar": "Use Bacillus thuringiensis (Bt) spray or handpick caterpillars.",
    "earthworms": "Earthworms are beneficial and should not be removed.",
    "earwig": "Use oil traps or remove debris from the field.",
    "grasshopper": "Use insecticides like Nosema locustae or natural predators.",
    "moth": "Install pheromone traps or use biological pesticides.",
    "slug": "Use iron phosphate baits or copper tape barriers.",
    "snail": "Apply mulch or use natural predators like ducks.",
    "wasp": "Remove nests carefully; use wasp traps if necessary.",
    "weevil": "Use insecticides or heat treatment for stored grains."
}
def train_environment_model():
    data = pd.read_csv("D:/Intern/environment_data.csv")
    X = data[["temperature", "humidity", "crop_type"]]
    y = data["infestation_risk"]
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=y)
    class_weight_dict = {i: class_weights[i] for i in range(3)}
    model = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight=class_weight_dict, random_state=42)
    model.fit(X, y)
    joblib.dump(model, "environment_model.pkl")
    return model

st.title("🌱 Pest and Environmental Risk Prediction Tool")
uploaded_image = st.sidebar.file_uploader("📸 Upload a Pest Image", type=["jpg", "jpeg", "png"])
temperature = st.sidebar.slider("🌡️ Temperature (°C)", 10, 40, 25)
humidity = st.sidebar.slider("💧 Humidity (%)", 20, 100, 60)
crop_type = st.sidebar.selectbox("🌾 Crop Type", ["Wheat", "Rice", "Corn", "Soybean"])
crop_mapping = {"Wheat": 0, "Rice": 1, "Corn": 2, "Soybean": 3}

if uploaded_image:
    try:
        pest_model = tf.keras.models.load_model("pest_classification_model_v2.h5")
        image = Image.open(uploaded_image).convert("L").resize((80, 80))
        img_array = np.array(image) / 255.0  
        img_array = img_array.reshape(1, 80, 80, 1)  
        label = np.argmax(pest_model.predict(img_array))
        pest_name = categories[label]
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.write(f"🌟 **Predicted Pest:** {pest_name}")
        st.write(f"🔧 **Control Recommendation:** {pest_recommendations.get(pest_name, 'No recommendation available for this pest.')}")
    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")

if st.sidebar.button("🔍 Predict Environmental Risk"):
    try:
        env_model = joblib.load("environment_model.pkl")
        features = pd.DataFrame([[temperature, humidity, crop_mapping[crop_type]]], columns=["temperature", "humidity", "crop_type"])
        risk = env_model.predict(features)[0]
        risk_messages = ["Low Risk", "Moderate Risk", "High Risk"]
        st.write(f"⚠️ **Environmental Risk:** {risk_messages[risk]}")

        if uploaded_image:
            st.write(f"🌾 **Combined Risk for {pest_name}:** {risk_messages[risk]} with suggested control.")
    except FileNotFoundError:
        st.error("The environmental model file (`environment_model.pkl`) was not found. Please train the model first.")
    except Exception as e:
        st.error(f"An error occurred during risk prediction: {e}")
