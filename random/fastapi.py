from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import uuid
import os

app = FastAPI()

# In-memory storage for models and their metadata
models = {}

# Model metadata structure
class ModelMetadata(BaseModel):
    id: str
    status: str
    accuracy: Optional[float]
    details: Dict[str, str]

# Request format for model details
class ModelDetailRequest(BaseModel):
    model_id: str

# Background task for model training
def train_model(model_id: str, file_path: str):
    # Simulate model training and store metadata
    try:
        # Simulate loading CSV and training (dummy process)
        df = pd.read_csv(file_path)
        # Simulate model training process
        accuracy = 0.9  # Let's assume the model training results in 90% accuracy
        
        models[model_id]["status"] = "done"
        models[model_id]["accuracy"] = accuracy
    finally:
        # Clean up: remove uploaded CSV file
        os.remove(file_path)

# 1. API for training a model
@app.post("/train/")
async def train_model_api(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    model_id = str(uuid.uuid4())
    file_location = f"temp_{model_id}.csv"
    
    # Save uploaded CSV file
    with open(file_location, "wb") as f:
        f.write(await file.read())
    
    # Create metadata for this model
    models[model_id] = {
        "id": model_id,
        "status": "in progress",
        "accuracy": None,
        "details": {}
    }

    # Start the training in the background
    background_tasks.add_task(train_model, model_id, file_location)
    
    return {"model_id": model_id}

# 2. API for prediction
@app.post("/predict/")
async def predict(file: UploadFile = None, input_data: Dict = None):
    if file:
        df = pd.read_csv(await file.read())
        predictions = df.applymap(lambda x: "predicted_value")  # Simulated prediction process
        response_file = "predictions.csv"
        predictions.to_csv(response_file, index=False)
        return {"prediction_file": response_file}
    
    elif input_data:
        predictions = {key: "predicted_value" for key in input_data}  # Simulated prediction
        return {"predictions": predictions}
    
    else:
        raise HTTPException(status_code=400, detail="Provide either a file or input_data")

# 3. API to list available models
@app.get("/models/")
async def list_models():
    return [{"model_id": model_id, "status": model["status"], "accuracy": model["accuracy"]} for model_id, model in models.items()]

# 4. API to get details about a particular model
@app.post("/model_details/")
async def model_details(request: ModelDetailRequest):
    model_id = request.model_id
    if model_id in models:
        return models[model_id]
    else:
        raise HTTPException(status_code=404, detail="Model not found")

# 5. API for health check
@app.get("/health/")
async def health_check():
    return {"status": "ok"}
    