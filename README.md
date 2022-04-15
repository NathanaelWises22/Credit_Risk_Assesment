# Credit_Risk_Assesment
 
From the dataset We are trying to create a model that can determine whether a customer will default or not. So that it can minimize risk for lenders and customers.
We achieve this by doing Data cleaning, simple EDA and Modeling using PyCaret library to create the Machine Learning model.

An 93,83% accuracy level was achieved in predicting the loan defaults on 32,576 loans and 12 benchmarks. With this model, the default rate would decrease by 15,83%, resulting in minimized risk for both the lender and applicant.

# Installation
```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import sklearn
import math

```
We'll be using PyCaret library to quickly create and compare models
```
pip install pycaret[full]
from pycaret.classification import setup
from pycaret.classification import compare_models
```
# Getting Started
```
df = pd.read_csv('credit_risk_dataset.csv')
df.head(5)
```
There's 4 categorical data and 8 Numerical Data.
32581 loans and 12 Features 
 ```
 df.info()
df.shape
```
# Data Pre-Processing 
We check for any NUll or Duplicated data and delete them.
```
#cek null data
df.isnull().sum()
 ```
 There's null Data "person_emp_length"  and "loan_int_rate" so we fill them using the median of the feature.
 ```
 df["person_emp_length"].fillna(df["person_emp_length"].median(), inplace=True)
df["loan_int_rate"].fillna(df["loan_int_rate"].median(), inplace=True)
```
Checking and deleting any duplicated data
```
df.duplicated().sum()
df.drop_duplicates(inplace=True)
```
We check for any outlier on the Numerical Variable
```
# get the numberical variables
num_cols = pd.DataFrame(df[df.select_dtypes(include=['float', 'int']).columns])

num_cols_hist = num_cols.drop(['loan_status'], axis=1)
# visualize the distribution for each varieble
plt.figure(figsize=(12,16))

for i, col in enumerate(num_cols_hist.columns):
    idx = int('42'+ str(i+1))
    plt.subplot(idx)
    sns.distplot(num_cols_hist[col], color='forestgreen', 
                 kde_kws={'color': 'indianred', 'lw': 2, 'label': 'KDE'})
    plt.title(col+' distribution', fontsize=14)
    plt.ylabel('Probablity', fontsize=12)
    plt.xlabel(col, fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(['KDE'], prop={"size":12})

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.35,
                    wspace=0.35)
plt.show()
#before removing outlier
```
![credit risk dataset-before removing outlier](https://user-images.githubusercontent.com/92627169/163572104-80f73660-4319-4bc7-826a-d0b7cb891001.png)

Before removing the outlier, the data is positively skewed. So we check and remove any outlier so the data distributed more evenly.
We will check the outliers using IQR method.
```
#checking for outlier
def find_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]

likely_to_have_outlier = ["person_age","person_income","person_emp_length","cb_person_cred_hist_length","loan_amnt","loan_int_rate", "loan_percent_income"]
for col in likely_to_have_outlier:
    print(find_outliers(df, col).shape)
    
#function to remove outlier
def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_range = Q1 - 1.5*IQR
    upper_range = Q3 + 1.5*IQR
    df[col] = np.where(df[col]>upper_range, upper_range, df[col])
    df[col] = np.where(df[col]<lower_range, lower_range, df[col])
    return df
#clean the data from outlier
for col in likely_to_have_outlier:
    remove_outliers(df, col)
```

```
#get the cleaned numerical & categorical variable 
num_cols2 = pd.DataFrame(df[df.select_dtypes(include=['float', 'int']).columns])
# print the numerical variebles
num_cols2.columns

#get the categorical variable
cat_cols = pd.DataFrame(df[df.select_dtypes(include=['object']).columns])
cat_cols.columns
```
We visualize the data after removing any outlier
```
num_cols_hist = num_cols2.drop(['loan_status'], axis=1)
# visualize the distribution for each variable
plt.figure(figsize=(12,16))

for i, col in enumerate(num_cols_hist.columns):
    idx = int('42'+ str(i+1))
    plt.subplot(idx)
    sns.distplot(num_cols_hist[col], color='forestgreen', 
                 kde_kws={'color': 'indianred', 'lw': 2, 'label': 'KDE'})
    plt.title(col+' distribution', fontsize=14)
    plt.ylabel('Probablity', fontsize=12)
    plt.xlabel(col, fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(['KDE'], prop={"size":12})

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.35,
                    wspace=0.35)
plt.show()
#after removing outlier
```
![credit risk dataset-after removing outlier](https://user-images.githubusercontent.com/92627169/163572168-0f6e944a-bda1-436f-8586-0dc36656494a.png)


# Simple EDA
 We check the comaparison between the 2 classes of our target column to make sure there isn't any imbalance
 
 ```
#compare the loan status
df['loan_status'].value_counts()
ax = df['loan_status'].value_counts().plot(kind='bar', figsize=(6, 8), fontsize=13)
ax.set_ylabel("Number of Customer", fontsize=14);

totals = []
for i in ax.patches:
    totals.append(i.get_height())

total = sum(totals)

for i in ax.patches:
    ax.text(i.get_x() - .01, i.get_height() + .5, \
            str(round((i.get_height()/total)*100, 2))+'%', fontsize=15,
                color='#444444')
plt.show()
```
![loan status comparion](https://user-images.githubusercontent.com/92627169/163567405-c0cac97e-8e0f-459d-a215-0d8c0b8aa1d3.png)

Loan status which is our target column have 78:22 ratio of default and non-default.
Where 0 = non-default , 1 = default.
Because the data is a little bit imbalance, we will use fix that at the later stage

In the next bit, i create the pearson correlation between the numerical variable to 'Loan_status' and categorical variables to 'loan_status' 

```
#correlation to loan status
corr = num_cols.corr().sort_values('loan_status', axis=1, ascending=False)
corr = corr.sort_values('loan_status', axis=0, ascending=True)
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask, k=1)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(8, 6))
    ax = sns.heatmap(corr, mask=mask, vmin=corr.loan_status.min(), 
                     vmax=corr.drop(['loan_status'], axis=0).loan_status.max(),
                     square=True, annot=True, fmt='.2f',
                     center=0, cmap='RdBu',annot_kws={"size": 12})
                     
 ```
![credit risk dataset- correlation between num_variable](https://user-images.githubusercontent.com/92627169/163572248-986b5610-4d56-4243-8f86-8888a03530b5.png)

From the correlation matrix above we know that person_income, person_emp_length, and person_age: has negative effect on loan_status being default which means the higher this variable are the less likely it will makes a loan go default.

loan_percent_income, loan_int_rate, and loan_amnt has positive effect on loan_status being default which means, the higher this variable are the more likely it will make a loan default.

```
#correlation between the catogorical variables
encoded_cat_cols = pd.get_dummies(char_col)
cat_cols_corr = pd.concat([encoded_cat_cols, df_clean1['loan_status']], axis=1)
corr = cat_cols_corr.corr().sort_values('loan_status', axis=1, ascending=False)
corr = corr.sort_values('loan_status', axis=0, ascending=True)
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask, k=1)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(16, 10))
    ax = sns.heatmap(corr, mask=mask, vmin=corr.loan_status.min(), 
                     vmax=corr.drop(['loan_status'], axis=0).loan_status.max(), 
                     square=True, annot=True, fmt='.2f',
                     center=0, cmap='RdBu',annot_kws={"size": 10})
```
![correlation between cat variable](https://user-images.githubusercontent.com/92627169/163572281-6f590d68-5c1b-456a-91ac-b9fc7ee070e2.png)

Loan grade A-B-C, loan intent venture-education-personal, history person never default, homeownership mortgage-own : has negative effect on loan_status being default which means the higher this variable are the less likely it will makes a loan go default.

loan grade G-F-Y-E-D,loan intent medical,homeimprove,debt, homeownership rent,other : has postive effect on loan_status being default which means, the higher this variable are the more likely it will make a loan default.

# Split the Dataset

Splitting the Dataset for 70:30. We will held back the 30% part as unseen data to test our model at the final stage.

```
#split test-train data
from sklearn.model_selection import train_test_split

training_data, testing_data = train_test_split(df_clean1, test_size=0.3, random_state=25)

print(f"No. of training examples: {training_data.shape[0]}")
print(f"No. of testing examples: {testing_data.shape[0]}")

# No. of training examples: 22691
# No. of testing examples: 9725
```

# Create & Train the model.

You can find documentation about PyCaret here  https://medium.com/analytics-vidhya/pycaret-101-for-beginners-27d9aefd34c5 or https://pycaret.gitbook.io/docs/

We create the environment for Pycaret first. We will fix the imbalance issue on this step too.
Take note that PyCaret function will automatically split the data into training-test data on its own too.

```
from pycaret.classification import *  

#pycaret to determine the model
grid = setup(data= training_data, target= 'loan_status',fix_imbalance=True) #fix_imbalance will automaticaaly fix the imbalanced dataset by oversampling using the SMOTE method.

```
Create and compare the model

```
## evaluate models and compare models
best = compare_models()

# report the best model
print(best)
```
![PyCaret](https://user-images.githubusercontent.com/92627169/163572344-7b752ab3-f9f2-4e43-b4a1-e101ceebf473.png)

The best model for this dataset according to the PyCaret is CatBoost Classifier. Training Time-wise i think Light Gradient Boosting Machine	would be a better option since it gives a shorter training time and not-so-different accuracy and precision, but since I want the best in terms of Accuracy adn precision, this time i go with the Catboost Classifier model.
You can find the documentation about CatBoost Classifier here on https://catboost.ai/

to create the model, we call this function
```
#creating model
catboost = create_model('catboost')
#this model run on the PyCaret transformed training data set.
```
The result of our training model is accuracy 93,41%, prec 96,48%.
We can also plot the confusion matrix and classification report using the function below.

```
# Plotting the classification report
plot_model(catboost,plot='class_report')

# Plotting the confusion matrix
plot_model(catboost,plot='confusion_matrix')
#from the Confusion Matrix we can also know that at least there will be 1155 customers that we will reject because the model calculate they will almost certainly defaulting.
```
![CM sebelum tuning](https://user-images.githubusercontent.com/92627169/163572408-056f4b22-b54a-4a22-a479-acfd83ae15cf.png)


# HyperParameter Tuning
 To further increase the accuracy and precision of our model, we do HyperParameter Tuning. From Pycaret library,Hyperparameter tuning is quite simple. JUst put the function below. Take note however, by default it will automatically using the Random Grid Search method. 
```
 # tune model hyperparameters
tuned_cat = tune_model(catboost)

#evaluate tuned modedl
evaluate_model(tuned_cat)
```
![CM setelah tuning](https://user-images.githubusercontent.com/92627169/163572575-abf7fd1b-4aac-49fd-a915-ad3dbcfdf5b4.png)

After tuning, accuracy 92,90%, prec 93,41%. There's decrease on accuracy and precision. This may happen if the default parameter used on creating the model is actually better than the one we uso on HyperParameter Tuning.
So for now, We use the non-tuned model.

# Test The Model
We test the model on Transformed test data, the test data that PyCaret function automatically create,let's call it Evaluation Dataset for further on. Let's try to test our model, We only need to insert the function below to try it.

```
#test the model
predict_model(catboost)

final_cat = finalize_model(catboost)
print(final_cat)
```
From the Evaluation Dataset we get result Accuracy 93,38% & Precision 96,36%
Next, let's try to run it with the dataset we held back at the beginning. 

```
unseen_predictions = predict_model(catboost, data=testing_data)
unseen_predictions.head()
```

The Label and Score columns are added onto the data_unseen set. Label is the prediction and score is the probability of the prediction. Notice that predicted results are concatenated to the original dataset while all the transformations are automatically performed in the background.

```
!pip install pycaret-nightly
from pycaret.utils import check_metric
check_metric(unseen_predictions['loan_status'], unseen_predictions2['Label'], metric = 'Accuracy')
```
From the dataset that we held back, model perform with 93,83% accuracy

# Interpret the Model & Conclusion

display feature and their importance
```
plot_model(catboost, plot = 'feature')
```
![feature importance](https://user-images.githubusercontent.com/92627169/163572638-1c712a0a-6589-4b50-9048-599fa119ad8e.png)

5 most important feature are 1. loan_percent_income, 2. person_income, 3.person_home_ownership_rent, 4.loan_intent_VENTURE, 5.loan grade A

Conclusion : The 5 features that most affect the default of this model are loan_percent_income, person_income, person_home_ownership_rent, loan_intent_VENTURE, loan grade A .
From this dataset, we can reduce the percentage of customers who default at the beginning of 22% to 6.17% using the model we developed using the Catboost Classifier and perform with a level of Precision 96.91% & accuracy 93.83%.
With this 15.83% increase in the default rate, lenders and borrowers can be better protected from risk.
