Capstone1---ML_Zoom_Camp

This project is for the Capstone 1 project of the ML_Zoom_Camp, I have decided to use a dataset about Whiskys in order to predict if the Whisky has a Tobacco essence on it according to other variables.
I have tested many models, specifically: Xgboost, NaiveBayes and Decision Tree. However, I choose the Naive Bayes with a model acurracy near 85% to test something new and see how is in production.

The variables are:

All variables are integers

Body = [0-4]
Sweetness = [1-4]
Smoky = [0-4]
Medicinal = [0-4]
Honey = [0-4]
Spicy = [0-3]
Winey = [0-4]
Nutty = [0-4]
Malty = [0-3]
Fruity = [0-3]
Floral = [0-4]

DEVELOP OF THE PROJECT

Dataset:

whisky.csv

Scripts:

The project has 3 scripts/notebook.

Whisky_EDA.ipynb = The document that has the cleaning, EDA process and model selection

train.py = Is the script of the final model and the parameters of it including the model downloading

service.py = The script where is located the Bentoml service

Bentofile.yaml = The Bentofile of the Bentomodel

Other Files: Also, there are other files:

Pipfile

Pipfile.lock

README.md

GCP link to test the model = https://whisky-model-2ytgd45kfa-uc.a.run.app/#/Service%20APIs/whisky_pred__classify

## TESTING THE MODEL PICTURE

![Screenshot 2022-12-12 at 18 32 48](https://user-images.githubusercontent.com/98197260/207197743-4589412d-1bf1-4994-86bb-85a25e962fa7.png)




