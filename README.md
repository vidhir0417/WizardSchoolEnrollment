# Wizardry School Enrollment - Enrollment Predictions

This document details the project's core objective is to predict the enrollment of magic students into an esteemed Wizardry School.

## Purpose

The main purpose of this project is to:
* Anticipate the enrollment of magic students into the Wizardry School based on hidden patterns in a provided dataset.
* Develop a robust and flexible predictive model using various machine learning techniques.
* Provide reliable forecasts to assist the Wizardry School in its admission decisions.

## Dataset

The project utilizes a dataset specifically designed for student enrollment predictions at the Wizardry School.
* **Composition:** A sample dataset methodically arranged into training and test sets.
* **Size:** Approximately 700 observations in the training set and 200 in the test set.
* **Features:** Both sets consist of 11 descriptive features. The training set includes an additional outcome feature: "Admitted in School."
* **Key Observation:** The outcome variable ("Admitted in School") is slightly unbalanced, with more students not admitted than admitted.

## Methodology

The project followed a structured machine learning pipeline:

### 1. Data Understanding and Exploration
* Used Pandas' `.info()` and `.describe()` to check for missing values, data types, distributions and potential outliers.
* Visualized categorical variable distributions and correlations (e.g., Student Family & Student Siblings, Financial Background & Student Siblings, Admitted in School & Financial Background).
* Identified potential outliers in numerical variables (`Experience Level`, `Student Siblings`, `Student Family`, `Financial Background`).

### 2. Data Preprocessing
* **Data Splitting:** The training set was divided into a train set and a validation set using a 70/30 split.
* **Outlier Treatment:** Applied Winsorization to most numerical variables to restrict extreme values, except for "Financial Background" where a direct limit was applied due to its strong correlation with the outcome.
* **One-Hot Encoding:** Applied to four categorical variables (Program, Student Gender, School of Origin, Favorite Study Element) to prevent the model from inferring ordinal relationships.
* **Missing Value Imputation:**
    * Used KNN Imputer (with 7 neighbors) for "Experience Level."
    * "School Dormitory" was dropped due to 79% missing data and minimal correlation with the outcome.
* **Scaling:** Tested Min Max Scaler, Standard Scaler and Robust Scaler, with Robust Scaler being chosen for its performance.

### 3. Feature Selection
* Employed four different methods to determine significant features: Correlation, RFE (Recursive Feature Elimination), Lasso Regression and Random Forest.
* Variables showing weaker results were excluded from the final model.

### 4. Model Development and Selection
* **Model Type:** Focused on Boosting Ensemble Classifiers, specifically `AdaBoostClassifier()`, using a `DecisionTreeClassifier` as the base estimator.
* **Hyperparameter Tuning:** `GridSearch` was extensively used to find optimal hyperparameter combinations for various models.
* **Best Model:** The `AdaBoostClassifier()` with a `DecisionTreeClassifier` (max_depth=1, random_state=15) as the estimator, a learning rate of 0.8, and 100 estimators, was selected. It used the 'SAMME' algorithm.
* **Evaluation:** Models were evaluated based on F1 and Accuracy scores.
* **Overfitting Control:** The chosen model showed no significant overfitting (less than 1% difference between train and validation F1 scores).

## Results and Discussion

* **Final Model Performance:** The selected AdaBoost Classifier achieved an accuracy score of **81%** on the training set and **80%** on the validation set. It also yielded an F1-score of **84.21%** on the test data (public Kaggle score).
* **Dataset Imbalance:** The slight imbalance in the dataset (more rejections than acceptances) influenced predictions, leading the model to predict more acceptances. This is a recognized limitation given the context of a highly prestigious school.
* **Computational Complexity:** Acknowledged the computational cost due to extensive `GridSearch` across all parameters for all models.
* **Trade-offs:** Decisions were made to keep highly correlated variables (e.g., Gender Male and Female) and encode all 'n' categories rather than 'n-1' to prioritize accuracy, even if it went against conventional best practices for this type of project.

## Conclusion

The AdaBoost Classifier, with a Decision Tree estimator, proved to be the most accurate model with no overfitting for predicting student enrollment. Despite the dataset's slight imbalance and the computational intensity of the tuning process, the model offers reliable forecasts for the Wizardry School's admission decisions. Future work could involve using a more balanced dataset with more observations to further refine the model's predictive importance.

## References

* \[1\] Carina Albuquerque (2023), Handling Large Datasets.
* \[2\] GeeksforGeeks (2021), Winsorization.
* \[3\] Arthur Lamblet Vaz (2018), One-hot-encoding, o que é?
* \[4\] GeeksforGeeks (2023), Python Lambda Functions.
* \[5\] Carina Albuquerque (2023), Data Visualization with Matplotlib and Seaborn.
