🎓 Student Performance Prediction
This project predicts students' exam scores using a XGBRegressor Regression model. The workflow includes data exploration, preprocessing, model training, hyperparameter tuning, and deployment.

📌 Project Overview
The dataset (StudentPerformanceFactors.csv) contains multiple rows and columns, with features such as:

Numerical: Hours Studied, Attendance (%), Sleep Hours, Previous Scores, Family Income, etc.
Categorical: Parental Involvement, Motivation Level, Teacher Quality, Internet Access, and more.
The goal is to build a robust model to accurately predict students' exam scores based on these factors, enabling data-driven interventions for improved academic outcomes.

🛠 Project Workflow
Data Loading & Exploration

Import the dataset and examine its structure and key statistics.
Check for missing values and identify relationships between features and the target variable.
Preprocessing & Feature Engineering

Handle missing values by replacing categorical features with the mode and numerical features with the median.
Encode categorical variables using ordinal and binary encoding.
Scale numerical features to standardize their ranges.
Data Visualization & Analysis

Explore relationships using scatter plots, box plots, and heatmaps.
Analyze outliers and correlations to guide preprocessing and modeling decisions.
Model Implementation & Training

Split the data into training (80%) and testing (20%) sets.
Train a the model on the training set.
Hyperparameter Tuning

Model Evaluation

Assess the model using metrics like R², Mean Absolute Error (MAE), and Mean Squared Error (MSE).
Compare the model's performance with a baseline (DummyRegressor).
Deployment

Develop a REST API using FastAPI to expose the model as a prediction service.
Host the API on Render for public accessibility.

📂 Key Files
📜 ML-Individual.ipynb – Jupyter Notebook containing the complete workflow for EDA, preprocessing, and model training.
📊 StudentPerformanceFactors.csv – Dataset used for training and testing the model.
🖥 app.py – FastAPI backend for serving predictions via the deployed model.

🚀 How to Run
1.Install dependencies:
pip install fastapi uvicorn numpy pandas scikit-learn matplotlib seaborn joblib

2.Run the FastAPI app locally:
uvicorn app:app --reload

3.Access the API:
Open your browser and go to: http://127.0.0.1:8000/docs

4.Test the API:
Use the /predict endpoint to send a JSON payload with student data and receive the predicted exam score.

🌐 Live Deployment
The API is deployed on Render and accessible at:
https://ml-8-ztbs.onrender.com

📊 Model Performance
Best Model (Random Forest Regressor):

R² Score: 0.63
MAE: 1.17
MSE: 5.19

🔍 Insights
Factors like Hours_Studied, Motivation_Level, and Previous_Scores are strong predictors of student performance.
Extracurricular activities and teacher quality moderately influence exam outcomes.

🚀 Future Improvements
Collect additional features, such as study environment and parental support.
Experiment with advanced machine learning models like Gradient Boosting or Neural Networks.
Add monitoring and logging to track real-time performance in deployment.
