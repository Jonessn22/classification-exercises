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
EDA
- initial investigations
- discover patterns
- spot anomolies
- formulate and test hypothesis
- check assumptions
    - summary statistics
    - graphical representations

## 05  Model Evaluation
How we evaluate our classification model's performance

## 06  Modeling
####    06-01   Decision Tree
####    06-02   Random Forest
####    06-03   KNN
####    06-04   Logistic Regression

## 99 Imports
******ignore warnings
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
import os

******files/data
from pydataset import data
import env
import aquire
import prepare

******visualizations
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

******sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

np.random.seed(123)