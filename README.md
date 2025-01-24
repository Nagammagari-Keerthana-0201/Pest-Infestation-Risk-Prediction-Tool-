# Pest Infestation Risk Prediction Tool 

## Overview
This project is a machine learning-based application designed to:
- Predict the type of pest from an uploaded image.
- Assess the environmental risk of pest infestation based on temperature, humidity, and crop type.

The tool is built using **Streamlit** for the user interface and incorporates pre-trained machine learning models for pest classification and environmental risk prediction.

---

## Features

1. **Pest Image Classification**:
   - Upload a pest image (e.g., ants, bees, beetles, etc.).
   - The model predicts the pest category based on a convolutional neural network (CNN).

2. **Environmental Risk Prediction**:
   - Input environmental conditions like temperature, humidity, and crop type.
   - The model predicts the risk level (Low, Moderate, High) using a Random Forest Classifier.
3. **Environmental Model Training**:
   - Generate the environmental model (environment_model.pkl) by running the training function in the code.
   - Allows customization with your own dataset.


## Technologies Used

- **Python**
- **Streamlit** for the web interface
- **TensorFlow** for pest classification (CNN model)
- **Scikit-learn** for environmental risk prediction (Random Forest Classifier)
- **Pandas** and **NumPy** for data handling
- **Pillow** for image processing

## Folder Structure

project/
│
├── train_environment_model.py
├── pest_image_processing.py      
├── streamlite_app.py             
├── requirements.txt              
├── README.md                    
├── environment_data.csv          
├── pest_classification_model_v2.h5   
├── environment_model.pkl   
├── train/                        
│   ├── ants/
│   ├── bees/
│   ├── beetle/
│   ├── ...
```

## Installation

### Prerequisites
- Python 3.8 or higher
- Git installed on your system

### Steps
1. Clone the repository:
   
   git clone https://github.com/your-username/pest-risk-prediction.git
   cd pest-risk-prediction
   

2. Set up a virtual environment:
   
   python -m venv venv
   source venv\Scripts\activate
   

3. Install dependencies:
   
   pip install -r requirements.txt
   

4. Train the models (if needed):
   - Pest Classification Model:
     python pest_image_processing.py
     
   - Environmental Risk Model:
     python -c "from streamlite_app import train_environment_model; train_environment_model()"
   

5. Run the Streamlit application:
   streamlit run streamlite_app.py

---

## Usage

1. **Upload a Pest Image**:
   - Use the sidebar to upload an image of a pest.
   - The app will classify the pest and display the predicted category.

2. **Predict Environmental Risk**:
   - Adjust the temperature, humidity, and select the crop type in the sidebar.
   - Click the "Predict Environmental Risk" button to see the risk level.
3. **Combined Predictions**:
   - If both the pest image and environmental data are provided, the app combines the predictions to suggest actionable recommendations.


## Example

1. Upload an image of a pest (e.g., an ant):
   - The app predicts: `Ants`

2. Set environmental conditions:
   - Temperature: `30°C`
   - Humidity: `70%`
   - Crop Type: `Rice`

   Prediction: `High Risk`

---

## Requirements

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


## Acknowledgments

- **Streamlit** for making it easy to build interactive web apps.
- **TensorFlow** and **Scikit-learn** for providing powerful machine learning tools.
- The pest image dataset and environmental data used for model training.
