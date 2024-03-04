
# Blood Transfusion Analysis

## Overview

This project analyzes blood donation data to predict whether an individual donated blood in March 2007. It includes data preprocessing, model training with TPOT and Linear Regression, log normalization, and AUC score evaluation.

## Prerequisites

Ensure you have the required libraries installed:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tpot
```

## Code Highlights

### Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import roc_auc_score
from tpot import TPOTClassifier
```

### Load Dataset

```python
data = pd.read_csv("path/to/transfusion.csv")
```

### Data Preprocessing

```python
data.rename(columns={'whether he/she donated blood in March 2007': 'target'}, inplace=True)
X = data.drop(columns='target')
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
```

### Model Training with TPOT

```python
pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, scoring='roc_auc', random_state=42)
pipeline_optimizer.fit(X_train, y_train)
print(pipeline_optimizer.score(X_test, y_test))
pipeline_optimizer.export('tpot_exported_pipeline.py')
```

### Log Normalization

```python
col_to_normalize = 'Monetary (c.c. blood)'
X_train_normed, X_test_normed = X_train.copy(), X_test.copy()
for df_ in [X_train_normed, X_test_normed]:
    df_['monetary_log'] = np.log(df_[col_to_normalize])
    df_.drop(columns=col_to_normalize, inplace=True)
```

### Linear Regression Model

```python
lr = LinearRegression()
lr.fit(X_train_normed, y_train)
logreg_auc_score = roc_auc_score(y_test, lr.predict(X_test_normed))
print(f'AUC score: {logreg_auc_score:.4f}')
```

### Model Comparison

```python
model_scores = [('TPOT', pipeline_optimizer.score(X_test, y_test)), ('Linear Regression', logreg_auc_score)]
sorted_scores = sorted(model_scores, key=lambda x: x[1], reverse=True)
print('Model Comparison:')
for model, score in sorted_scores:
    print(f'{model}: {score:.4f}')
```

## Conclusion

This README provides a concise overview of the blood transfusion analysis project. Explore the code for details and insights.


