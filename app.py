from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load("student_performance_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define input model (MATCHING MODEL'S FEATURE ORDER)
class StudentPerformanceInput(BaseModel):
    Hours_Studied: float
    Attendance: float
    Parental_Involvement: int
    Access_to_Resources: int
    Extracurricular_Activities: int
    Sleep_Hours: float
    Previous_Scores: float
    Motivation_Level: int
    Internet_Access: int
    Tutoring_Sessions: float
    Family_Income: int
    Teacher_Quality: int
    School_Type: int
    Peer_Influence: int
    Physical_Activity: float
    Learning_Disabilities: int
    Parental_Education_Level: int
    Distance_from_Home: int
    Gender: int

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/style", StaticFiles(directory="style"), name="style")

@app.get("/")
async def read_root():
    return FileResponse("style/index.html")

@app.post("/predict")
async def predict_performance(input_data: StudentPerformanceInput):
    try:
        # Convert to DataFrame with EXACT TRAINING FEATURE ORDER
        input_dict = input_data.dict()
        
        # Critical: Match the feature order seen in the error message
        columns_order = [
            'Hours_Studied', 'Attendance', 'Parental_Involvement',
            'Access_to_Resources', 'Extracurricular_Activities',
            'Sleep_Hours', 'Previous_Scores', 'Motivation_Level',
            'Internet_Access', 'Tutoring_Sessions', 'Family_Income',
            'Teacher_Quality', 'School_Type', 'Peer_Influence',
            'Physical_Activity', 'Learning_Disabilities',
            'Parental_Education_Level', 'Distance_from_Home', 'Gender'
        ]
        
        input_df = pd.DataFrame([input_dict], columns=columns_order)
        
        # Scale numerical features (same as training)
        numerical_cols = ["Hours_Studied", "Sleep_Hours", "Previous_Scores", "Physical_Activity"]
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
        
        # Predict
        prediction = model.predict(input_df)
        
        return {"Predicted Exam Score": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")