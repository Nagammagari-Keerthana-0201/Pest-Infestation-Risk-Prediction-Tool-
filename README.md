# 🌟 Pest Infestation Risk Prediction Tool 🌟

## 📝Overview
This project is a machine learning-based application designed to:
1️⃣Predict the type of pest from an uploaded image🐛.
2️⃣Assess the environmental risk of pest infestation based on temperature, humidity, and crop type.

The tool is built using **Streamlit** for the user interface and incorporates pre-trained machine learning models for pest classification and environmental risk prediction.

---

## 🚀Features

1. **🐞Pest Image Classification**:
   - Upload a pest image (e.g., 🐜 ants, 🐝 bees, 🪲 beetles).
   - The model predicts the pest category based on a convolutional neural network (CNN).

2. **🌍Environmental Risk Prediction**:
   - Input 🌡️ temperature, 💧 humidity, and 🌾 crop type to get a risk level:
      🟢 Low, 🟠 Moderate, 🔴 High
   - The model predicts the risk level (Low, Moderate, High) using a Random Forest Classifier.
3. **🛠️Environmental Model Training**:
   - Generate the environmental model (environment_model.pkl) by running the training function in the code.
   - Allows customization with your own dataset.


## Technologies Used

- **🐍 Python**
- **🎨 Streamlit** for the web interface
- **🧠 TensorFlow** for pest classification (CNN model)
- **🌳 Scikit-learn** for environmental risk prediction (Random Forest Classifier)
- 📊 **Pandas** and **NumPy** for data handling
- **🖼️ Pillow** for image processing

## Installation

### Prerequisites 🛠️
- Python 3.8+ 🐍
- Git installed 💾

### Steps 🚶‍♂️
1️⃣ Clone the repository:
   
   git clone https://github.com/your-username/pest-risk-prediction.git
   cd pest-risk-prediction
   

2️⃣ Set up a virtual environment:
   
   python -m venv venv
   source venv\Scripts\activate
   

3️⃣ Install dependencies:
   
   pip install -r requirements.txt
   

4️⃣ Train the models (if needed):
   - Pest Classification Model🐛:
     python pest_image_processing.py
     
   - Environmental Risk Model🌍:
     python -c "from streamlite_app import train_environment_model; train_environment_model()"
   

5️⃣Run the Streamlit application:
   streamlit run streamlite_app.py

---

## 🔧 Usage

1️⃣ **Upload a Pest Image 🖼️**:
   - Use the sidebar to upload an image of a pest.
   - The app will classify the pest and display the predicted category.

2️⃣ **Predict Environmental Risk 🌍**:
   - Adjust 🌡️ temperature, 💧 humidity, and 🌾 crop type in the sidebar.
   - Click "🔍 Predict Environmental Risk" to view the risk level.
3️⃣ **Combined Predictions**:
   - If both the pest image and environmental data are provided, the app combines the predictions to suggest actionable recommendations.

4️⃣ Example:
   - Image: 🐜 Ant
   - Temperature: 30°C 🌡️
   - Humidity: 70% 💧
   - Crop: 🌾 Rice
   - Prediction:
     🐛 Pest: Ants
     🔴 Risk: High

## 📋Requirements

The `requirements.txt` file includes:
   streamlit
   tensorflow
   numpy
   pandas
   scikit-learn
   Pillow
   joblib

Install all dependencies using:
   - pip install -r requirements.txt


## 🙏 Acknowledgments

- **Streamlit** for making it easy to build interactive web apps.
- **TensorFlow** and **Scikit-learn** for providing powerful machine learning tools.
- The pest image dataset and environmental data used for model training.
