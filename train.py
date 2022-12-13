## LIBRARIES

## Librerias 
import pandas as pd
from numpy import mean
import pickle
import numpy as np 
import bentoml
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn import *
import sklearn as skl
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
import graphviz
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from numpy import std
from sklearn.preprocessing import scale 
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn import model_selection
from scipy import stats
from scipy.stats import boxcox 
import pylab as pl
from sklearn import linear_model
import pyreadr 
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import seaborn as sns 
import matplotlib as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import matplotlib.pyplot as plt
import scipy
from scipy.stats import skew
import xgboost as xgb

## Import datasets

por = pd.read_csv("./whisky.csv")
por
df = pd.DataFrame(por)


## Clean data set

def dropid(data):
    return data.drop(columns=['RowID','Latitude','Longitude','Postcode','Distillery'])

df = dropid(df)

df

## Create a dataframe with the predictor variables

categorical = ['Body', 'Sweetness', 'Smoky', 'Medicinal','Honey', 'Spicy',
       'Winey', 'Nutty', 'Malty', 'Fruity', 'Floral']


## CLEAN train data set

dv = DictVectorizer(sparse=False)
    
## Split data set 

df_train, df_test =train_test_split(df, test_size=0.30,random_state=123)

## Extracr Y variable
y_train = (df_train.Tobacco).values
y_test = (df_test.Tobacco).values

del df_train['Tobacco']
del df_test['Tobacco']

## To dictionary
dict_train = df_train.to_dict(orient='records')
dict_test = df_test.to_dict(orient='records')

## Vectorize

dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(dict_train)
X_test = dv.transform(dict_test)

## Naive Bayes

# Definir el modelo
gnb = GaussianNB()


# Fit el modelo
gnb.fit(X_train, y_train)

## ACCURACY

## Train data set
y_pred_train = gnb.predict(X_train)
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))


## Test data set
y_pred = gnb.predict(X_test)
print('Test Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

## Predictions
print(y_pred[0:10])

## ROC

roc = roc_auc_score(y_test, y_pred)
print("The Roc on Naive Bayes is:", roc)

## Save model with BentoML

bentoml.sklearn.save_model("Whisky_pred", gnb, 
                           custom_objects={
                               "DictVectorizer": dv
                           })

