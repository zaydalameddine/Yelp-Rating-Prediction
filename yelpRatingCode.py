import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#importing .json files filled with different information
businesses = pd.read_json('yelp_business.json',lines=True)
reviews = pd.read_json('yelp_review.json',lines=True)
users = pd.read_json('yelp_user.json',lines=True)
checkins = pd.read_json('yelp_checkin.json',lines=True)
tips = pd.read_json('yelp_tip.json',lines=True)
photos = pd.read_json('yelp_photo.json',lines=True)

#want to understand the data that i have in my dataframes better so I will adjust the num of col, num of chars is each col
pd.options.display.max_coloumns = 60
pd.options.display.max_colwidth = 500

businesses.head(5)
reviews.head(5)
users.head(5)
checkins.head()
tips.head(5)
photos.head(5)

# merging data from several files into single df on buisness_id
df = pd.merge(businesses, reviews, how='left', on='business_id')
df = pd.merge(df, users, how='left', on='business_id')
df = pd.merge(df, checkins, how='left', on='business_id')
df = pd.merge(df, tips, how='left', on='business_id')
df = pd.merge(df, photos, how='left', on='business_id')

# cleaning data to only include continuous or binary pieces of data
features_to_remove = ['address','attributes','business_id','categories','city','hours','is_open','latitude','longitude','name','neighborhood','postal_code','state','time']
df.drop(labels=features_to_remove, axis=1, inplace=True)

# checking to see id there are any missing values, causes issues with Linear Regression model. If true there is missing values
df.isna().any()

#filling missing values with 0
df.fillna({'weekday_checkins':0,
           'weekend_checkins':0,
           'average_tip_length':0,
           'number_tips':0,
           'average_caption_length':0,
           'number_pics':0},
          inplace=True)

#checking to see which variable has the highest correlation with the depedent variable stars
df.corr()

#creating new dataset which is a subset of the features with the strongest correlation to the ratings, can easily add more col if needed by adding them below
features = df[['average_review_length','average_review_age']]

# ratings have been split into their own dataset aswell
ratings = df['stars']

# spliting data randomly into 80% train and 20% test sets
x_train, x_test, y_train, y_test = train_test_split(features, ratings, test_size = 0.2, random_state = 1)

model = LinearRegression()
model.fit(x_train, y_train)

# using the .score() function to calculate the R^2 value to see how much variance there is in the dependant variable

#testing difference in value between training and testing sets
model.score(x_train, y_train)
model.score(x_test, y_test)

# predicting y vals based of x_test
y_predicted = model.predict(x_test)

#display the predicted data against the real data
plt.scatter(y_test,y_predicted)
plt.xlabel('Yelp Rating')
plt.ylabel('Predicted Yelp Rating')
plt.ylim(1,5)
plt.show()

# plotting is terrible so model needs to be heavily adjusted
#will try multiple different subsets of data to see which has the best model
# to do this will create a function that takes in the feature list and plots the model following the same steps as above found in file model_features_with_graphs

def model_features(feature_list):
    
    # define ratings and features, with the features limited to our chosen subset of data
    ratings = df.loc[:,'stars']
    features = df.loc[:,feature_list]
    
    # perform train, test, split on the data
    X_train, X_test, y_train, y_test = train_test_split(features, ratings, test_size = 0.2, random_state = 1)
    
    #allowing the data to in escence be useable
    if len(X_train.shape) < 2:
        X_train = np.array(X_train).reshape(-1,1)
        X_test = np.array(X_test).reshape(-1,1)
    
    # create and fit the model to the training data
    model = LinearRegression()
    model.fit(X_train,y_train)
    
    # print the train and test scores
    print('Train Score:', model.score(X_train,y_train))
    print('Test Score:', model.score(X_test,y_test))
    
    # print the model features and their corresponding coefficients, from most predictive to least predictive
    print(sorted(list(zip(feature_list,model.coef_)),key = lambda x: abs(x[1]),reverse=True))
    
    # calculate the predicted Yelp ratings from the test data
    y_predicted = model.predict(X_test)
    
    # plot the actual Yelp Ratings vs the predicted Yelp ratings for the test data
    plt.scatter(y_test,y_predicted)
    plt.xlabel('Yelp Rating')
    plt.ylabel('Predicted Yelp Rating')
    plt.ylim(1,5)
    plt.show()

# subset of only average review sentiment
sentiment = ['average_review_sentiment']
model_features(sentiment)

# subset of all features that have a response range [0,1]
binary_features = ['alcohol?','has_bike_parking','takes_credit_cards','good_for_kids','take_reservations','has_wifi']
model_features(binary_features)

# subset of all features that vary on a greater range than [0,1]
numeric_features = ['review_count','price_range','average_caption_length','number_pics','average_review_age','average_review_length','average_review_sentiment','number_funny_votes','number_cool_votes','number_useful_votes','average_tip_length','number_tips','average_number_friends','average_days_on_yelp','average_number_fans','average_review_count','average_number_years_elite','weekday_checkins','weekend_checkins']
model_features(numeric_features)

# all features
all_features = binary_features + numeric_features
model_features(all_features)