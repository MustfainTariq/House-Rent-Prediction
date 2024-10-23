# House Price Prediction - MLOps Pipeline

This project demonstrates an end-to-end MLOps pipeline for a house price prediction model. The system includes model development, API integration, and deployment with Continuous Integration and Continuous Deployment (CI/CD) practices using GitHub Actions, Vercel, Flask, and Docker.

## Table of Contents
1. [Overview](#overview)
2. [Project Setup](#project-setup)
3. [Model Training](#model-training)
4. [API Integration](#api-integration)
5. [Frontend Interface](#frontend-interface)
6. [CI/CD Pipeline](#cicd-pipeline)
7. [Multi-Environment Testing & Deployment](#multi-environment-testing--deployment)
8. [Technologies Used](#technologies-used)
9. [How to Run Locally](#how-to-run-locally)

## Overview

This project is part of the **MLOps Fall 2024** course. It involves building, testing, and deploying a machine learning model to predict house prices. The model is served through a Flask API, and users can interact with it via a simple frontend interface. The deployment process utilizes GitHub Actions for CI/CD and Vercel for hosting.

## Project Setup

The project is divided into four milestones:
- **Milestone 1**: Project Scaffolding and Branching
- **Milestone 2**: Data Preprocessing and Model Training
- **Milestone 3**: API Integration with Frontend
- **Milestone 4**: CI/CD with Multi-Environment Testing

### Repository Structure:
- `main.py`: Handles data preprocessing, model training, and prediction logic.
- `app.py`: The Flask API that exposes the `/predict` endpoint.
- `test.py`: Contains unit tests for the API and the machine learning model.
- `requirements.txt`: Lists dependencies needed to run the project.
- `Dockerfile`: Used for containerizing the app to ensure environment parity across dev, stage, and prod.
- `README.md`: This file!

## Model Training

We trained a house price prediction model using a real-world dataset from [Kaggle](https://www.kaggle.com/). The model was built with `scikit-learn` and optimized using hyperparameter tuning.

### Features:
- **Data Preprocessing**: Missing value handling, feature scaling.
- **Model**: Gradient Boosting model.
- **Evaluation**: RMSE, R² metrics to ensure model performance.

## API Integration

The trained model is exposed through a Flask API. The `/predict` endpoint accepts house-related data via POST requests and returns the predicted house price.

### Example API Request:
```bash
POST /predict
{
    "location": "F-8",
    "area": 2000,
    "bedrooms": 4,
    "baths": 3
}

