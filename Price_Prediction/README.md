**Input:** Unclean data, where for many samples some feature values are missing, and sometimes labels are missing. The data is historical data of fuel prices in switzerlands surrounding country and the labels are the respective swiss fuel prices on that season. 


**Output:** Given Data of surrounding countries fuel prices, the ouptut is the predicted swiss fuel price.

***How it is done:*** 
* Data is loaded as a pandas frame
* Data made cleaner using an Imputer.
* We shuffle the data since it is ordered timewise and we do not want that in the cross validation one fold gives for example a certain type of correlation even though this was only the case for this specific few years.
* used cross validation to test different models and different kernels and chose the best one manually.
* used the optimal model to predict the price of fuel in switzerland in francs
  
