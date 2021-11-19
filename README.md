# Methodologies I >> Classification
- a. Lesson Notes
- b. Exercises

## 01  Data Acquisition
Acquiring and importing the data we will be using
- aquire.py file

## 02  Data Preparation
Preparing and cleaning our imported data
- prepare.py file

## 03  Tidy Data
- data should be tabular (made up of rows and columns)
- there should only be one value per cell
- each variable should be one column
- each observation shpould be one row
<b>Melt</b> required when one variable is spread across multiple columns
<b>Pivot</b> required when one column contains multiple variables

## 04  Exploratory Analysis
EDA | In this step we determine which features to feed into our model
- initial investigations
- discover patterns
- spot anomolies
- formulate and test hypothesis
- check assumptions
    - summary statistics
    - graphical representations<br>
<p><b>X_train: </b>Feature variable columns, drop target variable column<br>
<b>y_train: </b>Series with our target variable column

## 05  Model Evaluation
How we evaluate our classification model's performance

## 06  Modeling
####    06-01   Decision Tree
####    06-02   Random Forest
####    06-03   KNN
####    06-04   Logistic Regression

## 99 Imports
******ignore warnings<br>
import warnings<br>
warnings.filterwarnings("ignore")<br><br>

import numpy as np<br>
import pandas as pd<br>
from scipy import stats<br>
import os<br><br>

******files/data<br>
from pydataset import data<br>
import env<br>
import acquire<br>
import prepare<br><br>

******visualizations<br>
import matplotlib.pyplot as plt<br>
%matplotlib inline<br>
import seaborn as sns<br><br>

******sklearn<br>
from sklearn.model_selection import train_test_split<br>
from sklearn.tree import DecisionTreeClassifier<br>
from sklearn.tree import export_graphviz<br>
from sklearn.metrics import classification_report<br>
from sklearn.metrics import confusion_matrix<br><br>

np.random.seed(123)