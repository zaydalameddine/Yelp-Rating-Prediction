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

# merging data from several files into single df on buisness_id
df = pd.merge(businesses, reviews, how='left', on='business_id')
df = pd.merge(df, users, how='left', on='business_id')
df = pd.merge(df, checkins, how='left', on='business_id')
df = pd.merge(df, tips, how='left', on='business_id')
df = pd.merge(df, photos, how='left', on='business_id')

#filling missing values with 0
df.fillna({'weekday_checkins':0,
           'weekend_checkins':0,
           'average_tip_length':0,
           'number_tips':0,
           'average_caption_length':0,
           'number_pics':0},
          inplace=True)

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

#Train Score: 0.6118980950438655
#Test Score: 0.6114021046919492
#[('average_review_sentiment', 2.3033908433749612)]

# subset of all features that have a response range [0,1]
binary_features = ['alcohol?','has_bike_parking','takes_credit_cards','good_for_kids','take_reservations','has_wifi']
model_features(binary_features)

#Train Score: 0.012223180709591164
#Test Score: 0.010119542202269072
#[('has_bike_parking', 0.19003008208038988), ('alcohol?', -0.14549670708138188), ('has_wifi', -0.1318739757776259), ('good_for_kids', -0.08632485990337416), ('takes_credit_cards', 0.071755364921953), ('take_reservations', 0.04526558530451624)]

# subset of all features that vary on a greater range than [0,1]
numeric_features = ['review_count','price_range','average_caption_length','number_pics','average_review_age','average_review_length','average_review_sentiment','number_funny_votes','number_cool_votes','number_useful_votes','average_tip_length','number_tips','average_number_friends','average_days_on_yelp','average_number_fans','average_review_count','average_number_years_elite','weekday_checkins','weekend_checkins']
model_features(numeric_features)

#Train Score: 0.6734992593766658
#Test Score: 0.671331879812014
#[('average_review_sentiment', 2.2721076642097016), ('price_range', -0.08046080962699616), ('average_number_years_elite', -0.07190366288054208), ('average_caption_length', -0.003347066007784957), ('number_pics', -0.00295650281289227), ('number_tips', -0.0015953050789025923), ('number_cool_votes', 0.0011468839227083013), ('average_number_fans', 0.0010510602097420108), ('average_review_length', -0.0005813655692093316), ('average_tip_length', -0.0005322032063458492), ('number_useful_votes', -0.00023203784758684865), ('average_review_count', -0.0002243170289504519), ('average_review_age', -0.00016930608165092445), ('average_days_on_yelp', 0.00012878025876722203), ('weekday_checkins', 5.9185807544599255e-05), ('weekend_checkins', -5.518176206975765e-05), ('average_number_friends', 4.826992111615165e-05), ('review_count', -3.483483763846468e-05), ('number_funny_votes', -7.884395674527804e-06)]

# all features
all_features = binary_features + numeric_features
model_features(all_features)

#Train Score: 0.6807828861895333
#Test Score: 0.6782129045869246
#[('average_review_sentiment', 2.2808456996623665), ('alcohol?', -0.14991498593470548), ('has_wifi', -0.1215538262926246), ('good_for_kids', -0.11807814422012669), ('price_range', -0.06486730150042035), ('average_number_years_elite', -0.06278939713895351), ('has_bike_parking', 0.02729696991228556), ('takes_credit_cards', 0.024451837853633578), ('take_reservations', 0.01413455917297467), ('number_pics', -0.0013133612300808058), ('average_number_fans', 0.0010267986822656092), ('number_cool_votes', 0.0009723722734405862), ('number_tips', -0.0008546563320874887), ('average_caption_length', -0.0006472749798197418), ('average_review_length', -0.000589625792027261), ('average_tip_length', -0.0004205217503403181), ('number_useful_votes', -0.00027150641256134763), ('average_review_count', -0.00023398356902511536), ('average_review_age', -0.00015776544111324202), ('average_days_on_yelp', 0.00012326147662882923), ('review_count', 0.00010112259377369408), ('weekend_checkins', -9.239617469627863e-05), ('weekday_checkins', 6.153909123135256e-05), ('number_funny_votes', 4.8479351024965417e-05), ('average_number_friends', 2.069584037374767e-05)]


#using random test values on model to predict rating on restarant

my_resturant = np.array([1,0,1,1,1,1, 31, 1, 3, 2, 1175, 596, 1, 16, 19, 43, 46, 6, 105, 2005, 12, 122, 1, 45, 50])
model.predict(danielles_delicious_delicacies)
