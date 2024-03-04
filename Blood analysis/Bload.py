#import nesesarry library

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor,VotingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from tpot import TPOTClassifier


#load dataset

data = pd.read_csv("C:/Users/furkan/Desktop/Yapay zeka/Projeler/Stores Sales/Blood analysis/transfusion.csv")
head = data.head()
info = data.info()
derscribe = data.describe()

# chance target result columns name to target
data.rename(
    columns={'whether he/she donated blood in March 2007': 'target'},
    inplace=True
)
data.head(2)

#Print target incidence proportions, rounding output to 3 decimal places
data.target.value_counts(normalize=True).round(3)

# Splitting the dataset into train and test


X = data.drop(columns='target')
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)



pipeline_optimizer = TPOTClassifier(
    generations=5,
    population_size=20,
    verbosity=2,
    scoring='roc_auc',
    random_state=42,
    disable_update_check=True,
    config_dict='TPOT light'
)

pipeline_optimizer.fit(X_train, y_train)
print(pipeline_optimizer.score(X_test, y_test))
pipeline_optimizer.export('tpot_exported_pipeline.py')

# AUC score for tpot model
tpot_auc_score = roc_auc_score(y_test, pipeline_optimizer.predict_proba(X_test)[:, 1])
print(f'\nAUC score: {tpot_auc_score:.4f}')

# Print best pipeline steps
print('\nBest pipeline steps:', end='\n')
for idx, (name, transform) in enumerate(pipeline_optimizer.fitted_pipeline_.steps, start=1):
    # Print idx and transform
    print(f'{idx}. {transform}')

#checking variance

# X_train's variance, rounding the output to 3 decimal places
X_train.var().round(3)

#log normalization
X_train_normed, X_test_normed = X_train.copy(), X_test.copy()
# Specify which column to normalize
col_to_normalize = 'Monetary (c.c. blood)'
# Log normalization of specified column
for df_ in [X_train_normed, X_test_normed]:
    # Add log normalized column
    df_['monetary_log'] = np.log(df_[col_to_normalize])
    # Drop the original column
    df_.drop(columns=col_to_normalize, inplace=True)

X_train_normed.var().round(3)

#import model

lr = LinearRegression()

# Train the model with log-normalized data
lr.fit(X_train_normed,y_train)
# AUC score for tpot model
logreg_auc_score = roc_auc_score(y_test, lr.predict(X_test_normed))
print(f'\nAUC score: {logreg_auc_score:.4f}')

# Importing itemgetter
from operator import itemgetter

# Sort models based on their AUC score from highest to lowest
sorted(
    [('tpot', tpot_auc_score), ('lr', logreg_auc_score)],
    key=itemgetter(1),
    reverse=True
)