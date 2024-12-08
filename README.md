# CSE151A_Group_Project


## 1. Introduction
For our project, we decided to use a Seoul bike rental dataset from the UC Irvine Machine Learning Repository. This dataset contains information about public bike rentals in Seoul from 2017 to 2018, along with weather and holiday information for each day within that timeframe. We chose this dataset because we thought trying to predict bike rental numbers using weather and holiday information was interesting, and we were curious to see if our assumptions–that bike rentals would be higher on days with good weather and on weekends or holidays–were correct. Plus, a few of us are bike enthusiasts, so this dataset seemed like a cool way to incorporate our interests into the project. 

In general, having a good predictive model is important because such models can be incredibly useful. At UCSD, for example, machine learning is being applied to all sorts of things, from signal recovery to analyzing bird calls. Additionally, a bad predictive model isn't just a nuisance, but can actually cause significant harm: as we've mentioned before in class, if a model incorrectly predicts that someone doesn't have cancer when they actually do, that mistake can cost the person their life. 

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
