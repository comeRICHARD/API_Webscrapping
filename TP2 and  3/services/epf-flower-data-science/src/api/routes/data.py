import json
import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter(
    prefix="/data",
    tags=["data"]
)

@router.get("/datasets")
async def get_datasets():
    try:
        # Load the datasets JSON file
        with open('src/config/urls.json', 'r') as file:
            datasets = json.load(file)
        return JSONResponse(content=datasets)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
