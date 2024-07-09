# Spaceship Titanic - Exploratory Data Analysis and Prediction Models

## Project Overview

The Spaceship Titanic dataset is part of a Kaggle competition that challenges participants to predict which passengers were transported to an alternate dimension during a journey through space. The goal of this project is to perform exploratory data analysis (EDA) and build machine learning models to predict the target variable.

## Dataset

The dataset can be found [here](https://www.kaggle.com/competitions/spaceship-titanic). It includes various features about passengers such as:

- **PassengerId**
- **HomePlanet**
- **CryoSleep**
- **Cabin**
- **Destination**
- **Age**
- **VIP**
- **RoomService**
- **FoodCourt**
- **ShoppingMall**
- **Spa**
- **VRDeck**
- **Name**
- **Transported** (target variable)

## EDA (Exploratory Data Analysis)

### 1. Loading the Data

- Load the dataset using pandas.
- Display the first few rows to understand its structure.

### 2. Data Cleaning

- Handle missing values.
- Convert categorical variables to numerical where necessary.

### 3. Data Visualization

- Visualize distributions of numerical features.
- Analyze relationships between features using correlation heatmaps.
- Explore categorical variables using bar charts.

### 4. Feature Engineering

- Create new features based on existing data (e.g., total spending).
- Normalize/scale numerical features if required.

## Machine Learning Models

### Logistic Regression

#### 1. Data Preparation

- Split the data into training and test sets.
- Normalize features for logistic regression.

#### 2. Model Training

- Train a logistic regression model on the training data.

#### 3. Model Evaluation

- Evaluate the model using accuracy and confusion matrix.

**Logistic Regression Results:**

- **Accuracy:** 0.7757
- **Classification Report:**

  accuracy 0.78

- **K-fold Cross Validation Score:**

- **Mean Accuracy:** 78.96%
- **Standard Deviation:** 1.22%

### Random Forest Classifier

#### 1. Data Preparation

- Split the data into training and test sets (if not already done).

#### 2. Model Training

- Train a Random Forest classifier on the training data.

#### 3. Model Evaluation

- Evaluate the model using accuracy and confusion matrix.

**Random Forest Results:**

- **Accuracy:** 0.7786
- **Classification Report:**

  accuracy 0.78

- **K-fold Cross Validation Score:**

- **Mean Accuracy:** 78.72%
- **Standard Deviation:** 0.91%

### Logistic Regression with Grid Search

**Best parameters:** `{'classifier__C': 10, 'classifier__solver': 'liblinear'}`  
**Best cross-validation score:** 0.79  
**Validation Accuracy:** 0.7757  
**Classification Report:**

accuracy 0.78

### Random Forest with Grid Search

**Best parameters:** `{'classifier__max_depth': 10, 'classifier__n_estimators': 200}`  
**Best cross-validation score:** 0.80  
**Validation Accuracy:** 0.7872  
**Classification Report:**

accuracy 0.79

## Results

- Both models performed similarly with Random Forest slightly outperforming Logistic Regression.
- Grid Search optimization improved the accuracy of both models, with the Random Forest model achieving the highest accuracy of 0.7872.
- Random Forest with Grid Search provided the best results, indicating it as the better model for this dataset.

## Requirements

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## How to Run

1. Clone the repository.
2. Install the required libraries.
3. Run the Jupyter notebook or Python script to see the analysis and model results.

## Author

- Harsh Khetan
