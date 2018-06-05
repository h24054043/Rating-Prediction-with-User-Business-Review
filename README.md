# Rating-Prediction-with-User-Business-Review
By the text review to predict the rate with user business review

# Overview 
 
# Package
pandas , numpy , sklearn 

# Data

training_data.csv , test_data.csv

# Vectorizer

TfidfVectorizer( max_df=0.8 , min_df=3 )

# Array

Include the user_id and the encoding text 

# Model and Parameters

Use the GridSearchCV select the parameters including the "C" and "gamma" 

Model: SVM (SVR),RandomForestClassifier , Logisitcal Regression , SGDRegressor

# Predict
The csv data includes the User_id and the stars
