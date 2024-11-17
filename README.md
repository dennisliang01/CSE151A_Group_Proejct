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

## Train your first model 
Because of simplicity our first choice was linear/polynomial regression. We tried up to degree 4. 
After plotting our fitting graph, we notice that the training MSE decreases as the polynomial degree increases. However, the testing MSE decreases from degree 1 to degree 2 and increases from degree 2 onwards. After degree 2, our model shows signs of overfitting. Therefore, we believe a second degree polynomial model fits our dataset the best.

Our first model can be found in [2_first_model.ipynb](/2_first_model.ipynb).

## Where does your model fit in the fitting graph? 
The degree 1 model is underfitting as both train as well as test error are high. Degree seems to fit well with a test error higher than the train error but both in acceptable range. Degree 3 and especially degree 4 show overfitting with a very high test error, while achieving a low training error.

## What are the next models you are thinking of and why? 
### Decision Trees 
Next, we would like to try decision trees. Many of the features seem easy to split. E.g. for the rainfall a split on no rain vs rain or rain, little rain (amount of mm yet to be decided), heavy rain. Also hour could be split to divide the day into night vs day, or smaller differentiation. 

## What is the conclusion of your 1st model? What can be done to possibly improve it? 
For a first approach the results are not too bad. A degree of two provides us with solid results. 
### Feature Expansion
Another approach would be feature expansion. 
We could create a new feature that represents an overall weather score by combining temperature, rainfall, humidity, wind speed and solar radiation, and dew point. Eventually sub-combinations like temperature and humidity or temperature and wind speed can be more beneficial. 
We might also create a feature whether it is a working day or a weekend. 
### Ridge or Lasso Regression 
Instead of simple polynomial regression exploring regularization by using Ridge or Lasso regression might also be a good idea. 
