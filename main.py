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

df_clean = df.drop_duplicates()
print("\n\nShape of the dataframe after removing duplicates:")
print(df_clean.shape)

print("\n\nFraud vs Non-Fraud cases:")
print(df_clean['Class'].value_counts())

print("\n\nSummary Statistics:")
print(df_clean.describe().T)

print("\n\nFraud vs Non-Fraud comparison:")
print(df_clean.groupby('Class').mean().T)

sns.countplot(x='Class', data=df_clean)
plt.title("Fraud vs Non-Fraud")
plt.show()

sns.boxplot(x='Class', y='Amount', data=df_clean)
plt.title("Amount vs Class")
plt.show()

corr = df_clean.corr()
class_corr = corr['Class'].sort_values(ascending=False)
print("\n\nCorrelation:")
print(class_corr)

plt.figure()
sns.heatmap(corr)
plt.title("Correlation Heatmap")
plt.show()

important_features = corr['Class'][abs(corr['Class']) > 0.15]
print("\n\nImportant features (|corr| > 0.15):")
print(important_features)

plt.figure()
sns.heatmap(important_features.to_frame())
plt.title("Important Features Heatmap")
plt.show()

features = df_clean.drop('Class', axis=1)
target = df_clean['Class']