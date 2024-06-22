# Spaceship Titanic - Exploratory Data Analysis and Prediction Models

## Project Overview

The Spaceship Titanic dataset is part of a Kaggle competition that challenges participants to predict which passengers were transported to an alternate dimension during a journey through space. The goal of this project is to perform exploratory data analysis (EDA) and build machine learning models to predict the target variable.

## Dataset

The dataset can be found [here](https://www.kaggle.com/competitions/spaceship-titanic). It includes various features about passengers such as:

- PassengerId
- HomePlanet
- CryoSleep
- Cabin
- Destination
- Age
- VIP
- RoomService
- FoodCourt
- ShoppingMall
- Spa
- VRDeck
- Name
- Transported (target variable)

## EDA (Exploratory Data Analysis)

1. **Loading the Data:**

   - Load the dataset using pandas.
   - Display the first few rows to understand its structure.

2. **Data Cleaning:**

   - Handle missing values.
   - Convert categorical variables to numerical where necessary.

3. **Data Visualization:**

   - Visualize distributions of numerical features.
   - Analyze relationships between features using correlation heatmaps.
   - Explore categorical variables using bar charts.

4. **Feature Engineering:**
   - Create new features based on existing data (e.g., total spending).
   - Normalize/scale numerical features if required.

## Machine Learning Models

### Logistic Regression

1. **Data Preparation:**

   - Split the data into training and test sets.
   - Normalize features for logistic regression.

2. **Model Training:**

   - Train a logistic regression model on the training data.

3. **Model Evaluation:**
   - Evaluate the model using accuracy, precision, recall, and F1-score.
   - Generate confusion matrix to visualize performance.

### Random Forest Classifier

1. **Data Preparation:**

   - Split the data into training and test sets (if not already done).

2. **Model Training:**

   - Train a Random Forest classifier on the training data.

3. **Model Evaluation:**
   - Evaluate the model using accuracy, precision, recall, and F1-score.
   - Generate feature importance plot to understand the impact of each feature.

## Results

- Summarize the results of both models.
- Compare the performance metrics of logistic regression and random forest classifier.
- Discuss which model performed better and why.

## Conclusion

- Summarize key findings from the EDA.
- Highlight the strengths and limitations of the predictive models.
- Suggest potential improvements and future work.

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
