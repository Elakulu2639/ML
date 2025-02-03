from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

# Define the input data model
class StudentPerformanceInput(BaseModel):
    Hours_Studied: float
    Attendance: float
    Sleep_Hours: float
    Previous_Scores: float
    Parental_Involvement: int
    Motivation_Level: int
    Teacher_Quality: int
    Internet_Access: int
    Extracurricular_Activities: int
    Family_Income: float
    Learning_Disabilities: int
    Physical_Activity: float
    Distance_from_Home: float
    Access_to_Resources: int
    Gender: int
    Parental_Education_Level: int
    Peer_Influence: int
    School_Type: int
    Tutoring_Sessions: float

# Load the trained model
try:
    model = joblib.load("student_performance_model.pkl")
except Exception as e:
    raise RuntimeError(f"Failed to load the model: {e}")

# Initialize the FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins; adjust for production
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict_performance(input_data: StudentPerformanceInput):
    try:
        # Convert input data to a NumPy array
        input_array = np.array([
            input_data.Hours_Studied,
            input_data.Attendance,
            input_data.Sleep_Hours,
            input_data.Previous_Scores,
            input_data.Parental_Involvement,
            input_data.Motivation_Level,
            input_data.Teacher_Quality,
            input_data.Internet_Access,
            input_data.Extracurricular_Activities,
            input_data.Family_Income,
            input_data.Learning_Disabilities,
            input_data.Physical_Activity,
            input_data.Distance_from_Home,
            input_data.Access_to_Resources,
            input_data.Gender,
            input_data.Parental_Education_Level,
            input_data.Peer_Influence,
            input_data.School_Type,
            input_data.Tutoring_Sessions
        ]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_array)

        # Return the prediction
        return {"Predicted Exam Score": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {e}")
