# CSE151A_Group_Project

Notebook Link: [https://colab.research.google.com/drive/1748EV2Sbf0I69aJNW2KFquCi2FpQOLMA?usp=sharing](https://colab.research.google.com/drive/1rqXG1tEI_nTDLgZxeTXwmq--Nhsp5cPs?usp=sharing)

# Milestone 2 

## Plans for Preprocessing
After exploring our data, we found out that we are not missing any data and there does not seem to be any abnormal data. Therefore, we do not plan on replacing any values or dropping any data. There are several columns in which we need to encode the data for future use.

Date:  We need to encode the date. The current format is a string, DD/MM/YYYY. As it is easier to work with numerical data, a possible encoding scheme is to convert the date to number of days from the starting date. However, this would lose some information such as the day of the week. We plan to apply min/max scaling, in order to compare the attributes against each other.  

Rented Bike Count: This is our target and we do not plan on making any changes to the data.

Hour, Temperature, Humidity, Wind Speed, Visibility, Dew Point Temperature, Solar Radiation, Rainfall, Snowfall: We will be scaling these features using min/max scaling. We will be doing this in order to compare the attributes against each other.

Seasons: The current format are in strings. We would be label encoding these data to a scale of 0 to 3. This will aid in our future analysis. As for the other feature, we will as well apply min/max scaling to the season features for the same reason. 

Holiday, Functioning Day: The data is in Yes/No format. Since this is essentially boolean data, we will be using label encoding to convert these data.

# Milestone 3 

## Finish Major Preprocessing 
We encoded the date as the date of the year, which is slightly different than the continuos numbers we originally proposed. However date of the year turned out to be easier to encode and we assume that this will also generalize better. For the other features we sticked to our originally ideas for encoding. 
Finally, we applied min-max-scaling.

All preprocessing can be found in [1_preprocessing.ipynb](/1_preprocessing.ipynb).

**Milestone 4 Update:** in accordance to the feedback we got on gradescope, we updated our scaling of the testing data and the scaling of our MSE. 

# Milestone 3 & 4 

## Methods 

### Train your first model 
Because of simplicity our first choice was linear/polynomial regression. We tried up to degree 4. 
After plotting our fitting graph, we notice that the training MSE decreases as the polynomial degree increases. However, the testing MSE decreases from degree 1 to degree 2 and increases from degree 2 onwards. After degree 2, our model shows signs of overfitting. Therefore, we believe a second degree polynomial model fits our dataset the best.

Our first model can be found in [2_first_model.ipynb](/2_first_model.ipynb).

### Train your second model 

As proposed in the last submission we implemented decision trees. We first tried the default settings of the DecisionTreeRegressor from sklearn. As warned in the documentation we got an extremely complex model with 13347 nodes and a depth of 32 which overfits our model heavily. 
Consequently, we limited the max_depth of our model and tried out parameters in the range from 1 to 20 for this. Limiting the max_depth is a simple way to keep the model smaller. We saw that a depth of 12 provided us with significantly better results. However the MSEs were still noticeably high. Thus, we decided to utilize hyperparameter tuning by making use of the grid search approach. Grid search is a basic, but relatively costly as exponential way of trying out hyperparameters as it is testing every possible combination of the hyperparameter values. As we do not have computation power issues yet we decided it is sufficient for our implementation.  
We adjusted the feature ranges until we had best results that were not at the boundary but in the middle of each interval. By that we ensured that there are no better values right outside of our interval. 

Our second model can be found in [3_second_model.ipynb](/3_second_model.ipynb).


## Conclusion 

### First model
[2_first_model.ipynb](/2_first_model.ipynb)

#### Where does your first model fit in the fitting graph? 
The degree 1 model is underfitting as both train as well as test error are high. Degree seems to fit well with a test error higher than the train error but both in acceptable range. Degree 3 and especially degree 4 show overfitting with a very high test error, while achieving a low training error.

#### What are the next models you are thinking of and why? 
##### Decision Trees 
Next, we would like to try decision trees. Many of the features seem easy to split. E.g. for the rainfall a split on no rain vs rain or rain, little rain (amount of mm yet to be decided), heavy rain. Also hour could be split to divide the day into night vs day, or smaller differentiation. 

#### What is the conclusion of your 1st model? What can be done to possibly improve it? 
For a first approach the results are not too bad. A degree of two provides us with solid results. 
##### Feature Expansion
Another approach would be feature expansion. 
We could create a new feature that represents an overall weather score by combining temperature, rainfall, humidity, wind speed and solar radiation, and dew point. Eventually sub-combinations like temperature and humidity or temperature and wind speed can be more beneficial. 
We might also create a feature whether it is a working day or a weekend. 
##### Ridge or Lasso Regression 
Instead of simple polynomial regression exploring regularization by using Ridge or Lasso regression might also be a good idea. 

### Second Model 
[3_second_model.ipynb](/3_second_model.ipynb)

#### Where does your second model fit in the fitting graph? 
With the hyperparameter tuning we ensured that we neither have a totally over- or under-fitting model. However, the MSE values are quite high compare to what we achieved with the polynomial regression model. 

#### What is the conclusion of your second model? What can be done to possibly improve it? 
Currently, the results are not satisfying compared to our initial polynomial regression model. 
Eventually, we should intensify our hyperparameter tuning considering the remaining parameters such as min_weight_fraction_leaf, max_features, max_leaf_nodes and ccp_alpha=0.0 which we have not specified yet. 
Manually preprocessing our data for the decision tree might also help. We could manually introduce thresholds for splitting the data. Another than that decision trees in comparison to polynomial regression tend to struggle more with identifying feature interaction consequently we should maybe reduce the number of features used in our model and e.g. use an overall weather score instead of the multiple weather patterns. This would also prevent our tree from becoming too large. 



# Milestone 5: final report

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
Overall, our project showed that predicting bike rentals using weather and holidy data is feasible with polynomial regression providing the best results among the models we implemented. While decision trees offered insights into data splits, they underperformed compared to polynomial regression and also increased the complexity of the model.

In the future, we would like to introduce new features such as a composite weather score to make it easier to predict bike rentals and improve our model perofmrance. We would also like to explore new models such as neural networks to see if they can improve our results. Our Hyperparameter tuning was limited by the time we had to work on the project, so we would like to explore more hyperparameter tuning in the future. Due to the exhaustive nature of hyperparameter tuninng, we also hope to utilize more computing resouces such as the San Diego Supercomputing Center.

Finally, we believe having more data would also help improve the accuracy of our model. During our initial project research, we found there were other bike sharing datasets from other countries around the world. The only challenge with incorporating multiple datasets is to ensure that the data is consistent and that the features are comparable.

