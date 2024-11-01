# CSE151A_Group_Project

Notebook Link: https://colab.research.google.com/drive/1748EV2Sbf0I69aJNW2KFquCi2FpQOLMA?usp=sharing

## Plans for Preprocessing
After exploring our data, we found out that we are not missing any data and there does not seem to be any abnormal data. Therefore, we do not plan on replacing any values or dropping any data. There are several columns in which we need to encode the data for future use.

Date:  We need to encode the date. The current format is a string, DD/MM/YYYY. As it is easier to work with numerical data, a possible encoding scheme is to convert the date to number of days from the starting date. However, this would lose some information such as the day of the week.

Rented Bike Count: This is our target and we do not plan on making any changes to the data.

Hour, Temperature, Humidity, Wind Speed, Visibility, Dew Point Temperature, Solar Radiation, Rainfall, Snowfall: We will be scaling these features using min/max scaling. We will be doing this in order to compare the attributes against each other.

Seasons: The current format are in strings. We would be label encoding these data to a scale of 0 to 3. This will aid in our future analysis.

Holiday, Functioning Day: The data is in Yes/No format. Since this is essentially boolean data, we will be using label encoding to convert these data.

