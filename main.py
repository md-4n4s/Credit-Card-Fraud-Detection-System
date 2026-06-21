import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

pd.set_option('display.width', 0)

df=pd.read_csv('creditcard.csv')

print("\nShape of the dataframe: ", df.shape)

print("\n\nData types of the dataframe:")
print(df.dtypes)

print("\n\nFirst 5 rows of the dataframe:")
print(df.head())

print("\n\nNull values of the dataframe:")
print(df.isnull().sum())

print("\n\nShape of the dataframe after removing duplicates:")
print(df.drop_duplicates().shape)

print("\n\nFraud vs Non-Fraud cases:")
print(df['Class'].value_counts())

print("\n\nSummary Statistics:")
print(df.describe().T)

print("\n\nFraud vs Non-Fraud comparison:")
print(df.groupby('Class').mean().T)