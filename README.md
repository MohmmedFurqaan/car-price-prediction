# car-price-prediction
this model is Built using Python and popular libraries like scikit-learn, pandas, and numpy, this project showcases how to preprocess data, train a model, and deploy a user-friendly interface for price predictions.

# Car Price Predictor


# Aim

This project aims to predict the Price of an used Car by taking it's Company name, it's Model name, Year of Purchase, and other parameters.


## How to use?

1. Clone the repository
2. Install the required packages

Some packages are:
 - numpy 
 - pandas 
 - scikit-learn
 - flask

3. Run the "application.py" file
And you are good to go. 

# Description

## What this project does?

1. This project takes the parameters of an used car like: Company name, Model name, Year of Purchase, Fuel Type and Number of Kilometers it has been driven.
   
2. It then predicts the possible price of the car. For example, the image below shows the predicted price of our Hyundai Grand i10. 



## How this project does?

1. First of all the data was scraped from Quikr.com (https://quikr.com) 
Link for data: 

2. The data was cleaned (it was super unclean :( ) and analysed.

3. Then a Linear Regression model was built on top of it which had 0.92 R2_score.

Link for notebook: 

4. This project was given the form of an website built on Flask where we used the Linear Regression model to perform predictions.

