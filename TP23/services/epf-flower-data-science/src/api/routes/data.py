import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import os
import json
from sklearn.model_selection import train_test_split
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from src.firestore import FirestoreClient  # Adjusted import for correct path


router = APIRouter(
    prefix="/data",
    tags=["data"]
)

@router.get("/iris")
async def get_iris_data():
    """
    Endpoint to return the Iris dataset as JSON.

    Returns:
        JSONResponse: Data from the Iris CSV file
    """
    try:
        # Define the path to the dataset
        dataset_path = "src/data/iris.csv"
        
        # Check if the dataset exists
        if not os.path.exists(dataset_path):
            raise HTTPException(status_code=404, detail="Iris dataset not found")

        # Load the dataset into a Pandas DataFrame
        df = pd.read_csv(dataset_path)

        # Convert the DataFrame to a JSON object
        data_json = df.to_dict(orient="records")

        return JSONResponse(content=data_json)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/crypto")
async def get_crypto_data():
    """
    Endpoint to return the Crypto dataset from a JSON file.

    Returns:
        JSONResponse: Data from the Crypto JSON file
    """
    try:
        # Define the path to the dataset
        dataset_path = "src/data/crypto.json"
        
        # Check if the dataset exists
        if not os.path.exists(dataset_path):
            raise HTTPException(status_code=404, detail="Crypto dataset not found")

        # Load the JSON file
        with open(dataset_path, 'r') as file:
            data = json.load(file)

        # Return the JSON response
        return JSONResponse(content=data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@router.get("/process")
async def process_iris_data():
    """
    Endpoint to process the Iris dataset for model training.
    
    Returns:
        JSONResponse: Processed Iris dataset
    """
    try:
        # Define the path to the dataset
        dataset_path = "src/data/iris.csv"
        
        # Check if the dataset exists
        if not os.path.exists(dataset_path):
            raise HTTPException(status_code=404, detail="Iris dataset not found")

        # Load the dataset into a Pandas DataFrame
        df = pd.read_csv(dataset_path)

        # Data processing steps:
        # 1. Check for missing values and handle them
        if df.isnull().values.any():
            df = df.dropna()  # Drop rows with any missing values as an example

        # 2. Encode categorical variables if needed (e.g., species column)
        if 'species' in df.columns:
            df['species'] = df['species'].astype('category').cat.codes  # Encode as integer codes

        # 3. Normalize or scale the features (e.g., Min-Max Scaling)
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        feature_columns = df.columns[:-1]  # Exclude the target column 'species'
        df[feature_columns] = scaler.fit_transform(df[feature_columns])

        # Convert the DataFrame to a JSON object
        processed_data_json = df.to_dict(orient="records")

        return JSONResponse(content=processed_data_json)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@router.get("/split")
async def split_iris_data():
    """
    Endpoint to split the Iris dataset into training and testing sets and save them as JSON files.

    Returns:
        JSONResponse: Training and testing sets for the Iris dataset
    """
    try:
        # Define the path to the dataset
        dataset_path = "src/data/iris.csv"
        
        # Check if the dataset exists
        if not os.path.exists(dataset_path):
            raise HTTPException(status_code=404, detail="Iris dataset not found")

        # Load the dataset into a Pandas DataFrame
        df = pd.read_csv(dataset_path)

        # Separate features and target variable
        X = df.drop(columns=['Species'])  # Features
        y = df['Species']  # Target

        # Split the dataset into training and testing sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Combine the training data into a dictionary
        train_data = {
            "X_train": X_train.to_dict(orient="records"),
            "y_train": y_train.tolist()
        }

        # Combine the test data into a dictionary
        test_data = {
            "X_test": X_test.to_dict(orient="records"),
            "y_test": y_test.tolist()
        }

        # Paths for saving JSON files
        train_file_path = "src/api/splited_data/train_data.json"
        test_file_path = "src/api/splited_data/test_data.json"

        # Save the training and test data as JSON files
        with open(train_file_path, 'w') as train_file:
            json.dump(train_data, train_file, indent=4)

        with open(test_file_path, 'w') as test_file:
            json.dump(test_data, test_file, indent=4)

        # Return the data in the response as well
        return JSONResponse(content={"train": train_data, "test": test_data})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@router.post("/train_model")
async def train_model():
    """
    Endpoint to train a classification model with the pre-processed Iris dataset and save it as a file.
    """
    try:
        # Define paths
        train_data_path = "src/api/splited_data/train_data.json"
        model_params_path = "src/config/model_parameters.json"
        model_save_path = "src/models/iris_model.pkl"

        if not os.path.exists(train_data_path):
            raise HTTPException(status_code=404, detail="Training data not found")

        # Load training data
        with open(train_data_path, 'r') as file:
            train_data = json.load(file)

        # Convert to DataFrame
        df = pd.DataFrame(train_data["X_train"])
        y_train = pd.Series(train_data["y_train"])

        # Drop 'Id' column if it exists
        if 'Id' in df.columns:
            df = df.drop(columns=['Id'])

        # Load model parameters
        if not os.path.exists(model_params_path):
            raise HTTPException(status_code=404, detail="Model parameters not found")

        with open(model_params_path, 'r') as file:
            model_params = json.load(file)

        # Initialize and train the model
        model = RandomForestClassifier(
            n_estimators=model_params["n_estimators"],
            criterion=model_params["criterion"],
            random_state=42
        )
        model.fit(df, y_train)

        # Save the model
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        joblib.dump(model, model_save_path)

        return JSONResponse(content={"message": "Model trained and saved successfully."})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

    
@router.post("/predict")
async def make_prediction(input_data: dict):
    """
    Endpoint to make predictions using the trained classification model.
    """
    try:
        # Path to the trained model
        model_path = "src/models/iris_model.pkl"
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model not found")

        # Load the model
        model = joblib.load(model_path)

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Drop 'Id' column if it exists in the input data
        if 'Id' in input_df.columns:
            input_df = input_df.drop(columns=['Id'])

        # Make prediction
        prediction = model.predict(input_df)

        response = {
            "predicted_class": prediction[0],
            "input_data": input_data
        }

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@router.post("/save_parameters")
async def save_model_parameters():
    """
    Endpoint to save model parameters to Firestore.
    """
    try:
        # Initialize Firestore client
        firestore_client = FirestoreClient()

        # Path to model parameters file
        model_params_path = "src/config/model_parameters.json"

        # Check if the model parameters file exists
        if not os.path.exists(model_params_path):
            raise HTTPException(status_code=404, detail="Model parameters file not found")

        # Load model parameters
        with open(model_params_path, 'r') as file:
            model_params = json.load(file)

        # Save parameters to Firestore
        collection_name = 'parameters'
        document_id = 'parameters'
        firestore_client.client.collection(collection_name).document(document_id).set(model_params)

        return JSONResponse(content={"message": "Parameters saved to Firestore successfully."})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")