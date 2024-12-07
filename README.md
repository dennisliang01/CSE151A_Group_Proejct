# CSE151A_Group_Project


## 1. Introduction

## 2. Methods
### 2.1 Data Exploration
During our initial exploration of the data we found that there was also no abnormal data that had to be replaced or dropped. However, we discovered that the date, seasons, and holiday columns needed to be encoded.

### 2.2 Preprocessing
We encoded the date as the date of the year. We encoded the seasons from strings to an integer from 0-3. Holidays were label encoded from a yes/no to a 1/0. All Attributes were scaled using min-max scaling. All preprocessing can be found in [1_preprocessing.ipynb](https://github.com/dennisliang01/CSE151A_Group_Project/blob/Milestone3/1_preprocessing.ipynb).

### 2.3 Model 1: Polynomial Regression
Our first model we tried was a linear regression model, and then a polynomial regression model with degrees from 2-4. For reasons we will discuss later, we decided to use degree 2 for our final polynomial regression model. Our first model can be found in [2_first_model.ipynb](https://github.com/dennisliang01/CSE151A_Group_Project/blob/Milestone3/2_first_model.ipynb).

### 2.4 Model 2: Decision Trees
Our second model was a decision tree. We used gridsearch to find the optimal hyperparameters. The optimal parameters used in our final model were: max_depth=20, min_samples_leaf=6, min_samples_split=24. Keep in mind we did not do an entirely exhaustive search, so there are other parameters that were not tuned. Our second model can be found in [3_second_model.ipynb](https://github.com/dennisliang01/CSE151A_Group_Project/blob/Milestone4/3_second_model.ipynb).


## 3. Results

## 4. Discussion

## 5. Conclusion
