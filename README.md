# Blood Donation Prediction

This project aims to predict a person's likelihood of donating blood using machine learning models. The dataset contains past blood donation information and other relevant features. The project involves data preprocessing, model training, and performance evaluation. Here are the steps and details of the project:

## Steps

1. **Import Necessary Libraries**: The required libraries for data analysis, model training, and evaluation are imported at the beginning of the project.

2. **Load the Dataset**: The project utilizes the `transfusion.csv` dataset, which includes information about blood donations.

3. **Data Preprocessing**: The dataset is transformed into a suitable format for model training and evaluation. Additionally, the name of the target variable is changed, and class balance is checked.

4. **Split the Dataset**: The dataset is divided into training and test sets to assess the model's generalizability.

5. **Model Selection**: The project offers two different model options, TPOT and a linear regression model (`LinearRegression`).

6. **Model Training**: The selected models are trained on the training dataset.

7. **Evaluate Model Performance**: The performance of the models is evaluated on the test dataset. Performance metrics include the AUC score and other accuracy metrics.

8. **Comparison of Results**: The performance of the models is compared, and the most suitable model is selected.

## Models Used

1. **TPOT (TBA)**: TPOT is employed for automatic model selection and hyperparameter tuning. TPOT searches a wide range of models and hyperparameters to find the best-performing model.

2. **Linear Regression**: Linear regression is chosen for its simplicity and speed. This model is trained on log-normalized data.

## Performance Evaluation

- AUC scores for the TPOT model and linear regression model are compared.
- The ranking of the models is determined based on their AUC scores.

This project demonstrates how machine learning models can be used to address blood donation prediction, with the performance of the models evaluated based on the dataset and application requirements.
