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
Before removing the outlier, the data should be positive skewed. So we check and remove any outlier so the data distributed more evenly.
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
Loan status which is our target column have 78:22 ratio of default and non-default.
Where 0 = non-default , 1 = default.
Because the data is a little bit imbalance, we will use fix that at the later stage

```
pip install ppscore

```
We use PPS Score to better understand the general correlation between the numerical and categorical variables.
I still use the pearson correlation method to understood the linear correlation between the variables.

```
plt.figure(figsize=(10,8))
matrix_df = pps.matrix(df)[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
sns.heatmap(matrix_df, vmin=0, vmax=1, cmap="Blues", linewidths=0.5, annot=True)
```


