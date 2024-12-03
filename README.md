# EPF-API-TP

***At the attention of **Mr Letizia**, i kept the TP2 and 3 just to have the history of the commits...
I changed the name of the folder due of the space in the name that was posing a problem so the real folder is TP23.***


## Overview
- **Purpose**: Build a REST API to train and predict using machine learning models, with data storage and retrieval from Firestore.

## Features
- **Data Endpoints**: Retrieve and process datasets (e.g., Iris dataset, Crypto data).
- **Model Training**: Train a RandomForest model on pre-processed data and save to disk.
- **Prediction**: Make predictions using the trained model.
- **Firestore Integration**: Store and retrieve model parameters using Firestore.
- **Data Splitting**: Split datasets into training and testing sets for model evaluation.

## Technologies Used
- **FastAPI**: Framework for building the RESTful API.
- **Scikit-Learn**: For data preprocessing and model training.
- **Pandas and JSON**: Data handling and transformation.
- **Firestore**: Cloud-based NoSQL database for storing model parameters.
- **Joblib**: Saving and loading trained models.

## Setup
1. Set up `GOOGLE_APPLICATION_CREDENTIALS` for Firestore.
2. Install dependencies via `pip install -r requirements.txt`.
3. Run the server with `uvicorn main:app --reload`.
