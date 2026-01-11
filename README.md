# Bank Marketing Term Deposit Prediction (MLOps)

An end-to-end MLOps pipeline for predicting customer subscriptions to bank term deposits. This project covers data preprocessing, model training, containerization with Docker, and cloud deployment.

##  Project Overview
The objective of this project is to build a predictive model to identify customers most likely to subscribe to a term deposit. By targeting the right audience, a bank can optimize its marketing campaigns and increase conversion rates.

##  Model and Data Pipeline
The following steps were implemented in the `model_training.ipynb` to develop the inference pipeline:

### 1. Data Ingestion & Cleaning
- **Source:** The `bank-additional.csv` dataset featuring 41,188 records and 21 variables.
- **Target Variable:** The `y` column (yes/no) was mapped to binary values (1/0).
- **Handling Imbalance:** Since only ~11% of customers subscribe, the model uses class-weight balancing to ensure fair prediction.

### 2. Preprocessing & Feature Engineering
A `scikit-learn` **ColumnTransformer** was used to create a reproducible pipeline:
- **Numerical Features:** Features like `age`, `duration`, and economic indicators (`euribor3m`, `nr.employed`) were processed using `StandardScaler` to normalize their scales.
- **Categorical Features:** Columns such as `job`, `marital`, and `education` were transformed using `OneHotEncoder` to create dummy variables.
- **Robustness:** The encoder is configured with `handle_unknown='ignore'` to prevent API crashes when encountering new categories during live inference.

### 3. Model Summary:

Algorithm: Logistic Regression with class_weight='balanced'. 
- Preprocessing: Data was processed using a ColumnTransformer that applied StandardScaler to numerical indicators (like euribor3m and nr.employed) and OneHotEncoder to categorical variables (like job and education).
- Justification: This model was chosen for its ability to handle the significant class imbalance in the bank marketing data and its low computational overhead, which is critical for real-time inference via a Dockerized Flask API.

### 4. Model Training
- **Algorithm:** Logistic Regression (Solver: `liblinear`).
- **Optimization:** Trained with `class_weight='balanced'` to improve sensitivity to the minority class (subscribers).
- **Performance:** Achieved an **AUC-ROC of 0.94**, indicating high discriminatory power between subscribers and non-subscribers.

##  Tech Stack
- **Languages:** Python 3.9
- **Libraries:** Pandas, Scikit-Learn, Flask, Joblib, Gunicorn
- **DevOps:** Docker
- **Cloud:** Azure / AWS (Linux EC2)

## Project Structure
- `model_training.ipynb`: Exploratory Data Analysis and Model Training code.
- `app.py`: Flask API script for serving the model.
- `final_model.pkl`: The serialized ML pipeline (Preprocessor + Model).
- `Dockerfile`: Instructions to build the Docker container.
- `requirements.txt`: Python dependencies.

## Deployment Instructions

### 1. Local Setup
```bash
# Clone and Install
git clone <your-repo-link>
cd <repo-folder>
pip install -r requirements.txt

# Build and Run Docker
docker build -t bank-marketing-model .
docker run -d -p 5000:5000 --name bank-predictor bank-marketing-model
```

### 2. Cloud Deployment
Provision a Linux VM on Azure/AWS.

Configure Security Groups to allow inbound traffic on Port 5000.

Install Docker on the VM, transfer the project files, and run the Docker commands above.

## Testing the API
Replace <PUBLIC_IP> with your cloud instance's IP address.

Endpoint: http://<PUBLIC_IP>:5000/predict

Method: POST

Example Request:
```bash
curl -X POST -H "Content-Type: application/json" \
-d '{"age": 30, "job": "blue-collar", "marital": "married", "education": "basic.9y", "default": "no", "housing": "yes", "loan": "no", "contact": "cellular", "month": "may", "day_of_week": "fri", "duration": 487, "campaign": 2, "pdays": 999, "previous": 0, "poutcome": "nonexistent", "emp.var.rate": -1.8, "cons.price.idx": 92.893, "cons.conf.idx": -46.2, "euribor3m": 1.313, "nr.employed": 5099.1}' \
http://<PUBLIC_IP>:5000/predict
```
Expected Response:
JSON

{
  "prediction": 0,
  "probability_yes": 0.031885994107234805
}

---

### Tips for your Video Demo:
When you record your video, be sure to highlight:
1.  **The Pipeline:** Mention that you didn't just save a model, but a **full pipeline** that includes preprocessing.
2.  **Imbalance:** Briefly mention using **balanced class weights**—this shows you understand the business problem of bank marketing.
3.  **Port 5000:** Show the API responding live from your cloud IP address.

