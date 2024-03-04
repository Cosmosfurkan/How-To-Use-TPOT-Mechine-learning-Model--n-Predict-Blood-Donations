# Blood Transfusion Analysis

## Overview

This project aims to analyze blood donation data using machine learning techniques. The dataset is loaded from a CSV file and processed to predict whether an individual donated blood in March 2007. The analysis includes data preprocessing, model training using TPOT (Tree-based Pipeline Optimization Tool) and Linear Regression, log normalization, and AUC score evaluation.

## Prerequisites

Before running the code, make sure you have the following libraries installed:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tpot

You can install them using the following command:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tpot
```

## Code Structure

The code is structured as follows:

1. **Import Necessary Libraries**
   - Import required libraries for data analysis and machine learning.

```python
# ... (import statements)
```

2. **Load Dataset**
   - Load the blood donation dataset from the specified file path.
   - Display basic information and statistics of the dataset.

```python
# ... (loading dataset and basic information)
```

3. **Data Preprocessing**
   - Rename the target column.
   - Display target incidence proportions.
   - Split the dataset into training and testing sets.

```python
# ... (data preprocessing)
```

4. **Model Training with TPOT**
   - Use TPOTClassifier for automated machine learning model selection and hyperparameter tuning.
   - Print AUC score for the TPOT model.
   - Export the best pipeline to a Python script.

```python
# ... (TPOT model training and evaluation)
```

5. **Log Normalization**
   - Check and normalize the variance of the training set.
   - Perform log normalization on the specified column.

```python
# ... (log normalization)
```

6. **Linear Regression Model**
   - Train a Linear Regression model using log-normalized data.
   - Print AUC score for the Linear Regression model.

```python
# ... (linear regression model training and evaluation)
```

7. **Model Comparison**
   - Sort models based on their AUC scores and display the results.

```python
# ... (model comparison)
```

## Conclusion

This README provides an overview of the blood transfusion analysis project. Feel free to explore the code for more details and insights.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
