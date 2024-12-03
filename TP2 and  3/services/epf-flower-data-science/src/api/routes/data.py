import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import os
import json


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

