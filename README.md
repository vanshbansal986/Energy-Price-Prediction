# Energy Price Prediction Project

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation and Setup](#installation-and-setup)
3. [Data Description](#data-description)
4. [Methodology](#methodology)
5. [Data Preprocessing](#data-preprocessing)
6. [Model Architecture](#model-architecture)
7. [Training Process](#training-process)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Results and Interpretation](#results-and-interpretation)
10. [Future Improvements](#future-improvements)

## Project Overview

This project aims to predict energy prices one day ahead using historical energy data. The project utilizes various machine learning models to forecast future energy prices based on a range of features including generation data, load forecasts, and temporal information.

## Installation and Setup

To run this project, you'll need Python 3.x and the following libraries:

```
matplotlib
numpy
pandas
scikit-learn
seaborn
scipy
ydata-profiling
```

You can install these libraries using pip:

```
pip install matplotlib numpy pandas scikit-learn seaborn scipy ydata-profiling
```

## Data Description

The project uses an energy dataset (`energy_dataset.csv`) which includes the following key features:

- Temporal data (Year, Month, Day, Hour)
- Energy generation data from various sources (e.g., fossil, nuclear, wind, solar)
- Load forecasts and actual loads
- Price data (actual and day-ahead predictions)

## Methodology

The project follows these main steps:

1. Data loading and initial exploration
2. Data preprocessing and cleaning
3. Feature engineering
4. Model selection and training
5. Model evaluation and comparison

## Data Preprocessing

1. **Date-time Conversion**: The 'time' column is converted to datetime format and split into separate columns (Year, Month, Day, Hour).

2. **Data Cleaning**:
   - Rows with null values in the 'time' column are dropped.
   - Columns with too many missing values or constant values are removed.
   - Remaining null values are dropped as they constitute less than 0.1% of the data.

3. **Feature Engineering**:
   - A 'predict' column is created by shifting the 'price actual' column by 24 hours, representing the target variable for next-day price prediction.

4. **Data Splitting**:
   - The data is split into training (70%) and testing (30%) sets.

## Model Architecture

The project implements a preprocessing pipeline followed by various regression models:

1. **Preprocessing Pipeline**:
   - Standard Scaling
   - Box-Cox Transformation (Power Transformer)
   - Principal Component Analysis (PCA) for feature selection

2. **Models Implemented**:
   - Random Forest Regressor
   - K-Nearest Neighbors Regressor
   - Linear Regression
   - Support Vector Regression (SVR)

## Training Process

Each model is trained using the following steps:

1. The preprocessing pipeline is applied to the input features.
2. The model is fitted on the preprocessed training data.
3. Hyperparameter tuning is performed using GridSearchCV for some models.

## Evaluation Metrics

The models are evaluated using the following metrics:

- R-squared (R2) Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Explained Variance Score (EVS)

Additionally, cross-validation scores are computed to assess model performance and generalization.

## Results and Interpretation

The performance of each model varies:

1. **Random Forest**:
   - Second best performing model with good R2 score and lowest error metrics.
   - Hyperparameter tuning further improved its performance.

2. **K-Nearest Neighbors (KNN)**:
   - Showed moderate performance.
   - Hyperparameter tuning helped in finding the optimal number of neighbors and weight function.

3. **Linear Regression**:
   - Performed average, indicating less linear relationship between most features and the target variable.

4. **Support Vector Regression (SVR)**:
   - Showed best performance after hyperparameter tuning.
   - The linear kernel was found to be most effective.

## Future Improvements

1. Feature Engineering: Create more sophisticated features that capture market trends and external factors affecting energy prices.

2. Ensemble Methods: Implement ensemble techniques to combine predictions from multiple models.

3. Time Series Specific Models: Explore models designed for time series data, such as ARIMA or Prophet.

4. Deep Learning: Investigate the use of neural networks, particularly recurrent neural networks (RNNs) or long short-term memory (LSTM) networks for sequence prediction.

5. External Data Integration: Incorporate external data sources such as weather forecasts or economic indicators that might influence energy prices.

6. Regularization Techniques: Apply regularization methods to prevent overfitting, especially for linear models.

7. Automated Machine Learning: Utilize AutoML tools to explore a wider range of models and hyperparameters.

8. Interpretability: Implement model interpretation techniques to understand feature importance and model decisions.

By continuously refining the model and incorporating these improvements, the accuracy and reliability of the energy price predictions can be enhanced over time.
