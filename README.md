# Yelp Rating Prediction - model_features_with_graphs.py is what you probably want to check out
 
In this project, I have taken data from Yelp in order to use multiple linear regression to predict the rating of a restuarant given other features about it

yelpRatingCode.py was my initial script where I created a model based on a small subset of the dataframe. 

However in model_features_with_graphs.py I create multiple models based on different subsets of data to create the best fitting model. In this file I also have written the code to print the graphs but to see the graphs, you would need to run the program on your computer.

I also:
- Used the pandas python library to create dataframes which held the data from the .csv files
- Normalized the data and plotted the independent variables against the dependent
- Compared different coefficient values to see which variables held the most weight on the dependent variable
- Split the data into training (80%) and testing (20%) sets
- Evaluated the accuracy of the model by calculating the coefficient of the determinant of R2

P.S. Data files are over 100mb, figuring out how to upload them, will be done shortly
