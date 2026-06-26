# Bike Sharing Demand Prediction

A machine learning project that predicts hourly bike rental demand using regression techniques and feature engineering. The project implements multiple regression models, applies data preprocessing, and compares their performance using Mean Squared Error (MSE) and R² score.

---

## Features

- Data preprocessing and feature engineering
- Feature scaling and one-hot encoding
- Linear Regression implementation
- Polynomial Regression (Degree 2, 3, and 4)
- Quadratic Regression with interaction features
- Performance evaluation using MSE and R² score
- Comparative analysis of regression models

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib

---

## Project Structure

- `bike_sharing_regression.py` – Complete machine learning pipeline
- `train.csv` – Bike Sharing Demand dataset
- `requirements.txt` – Python dependencies

---

## Dataset

This project uses the **Bike Sharing Demand** dataset from Kaggle.

The dataset contains historical information such as season, weather, temperature, humidity, wind speed, holidays, and working days to predict hourly bike rental demand.

---

## Models Implemented

- Linear Regression (Baseline)
- Polynomial Regression (Degree 2)
- Polynomial Regression (Degree 3)
- Polynomial Regression (Degree 4)
- Quadratic Regression with Interaction Features

---

## Evaluation Metrics

The implemented models are evaluated using:

- Mean Squared Error (MSE)
- R² Score

---

## How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python bike_sharing_regression.py
```
