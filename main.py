import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.width', 0)

df=pd.read_csv('creditcard.csv')

# print("\nShape of the dataframe: ", df.shape)

# print("\n\nData types of the dataframe:")
# print(df.dtypes)

# print("\n\nFirst 5 rows of the dataframe:")
# print(df.head())

df= df.drop('Time', axis=1)

# print("\n\nNull values of the dataframe:")
# print(df.isnull().sum())

df_clean = df.drop_duplicates()
# print("\n\nShape of the dataframe after removing duplicates:")
# print(df_clean.shape)

# print("\n\nFraud vs Non-Fraud cases:")
# print(df_clean['Class'].value_counts())

# print("\n\nSummary Statistics:")
# print(df_clean.describe().T)

# print("\n\nFraud vs Non-Fraud comparison:")
# print(df_clean.groupby('Class').mean().T)

# sns.countplot(x='Class', data=df_clean)
# plt.title("Fraud vs Non-Fraud")
# plt.show()

# sns.boxplot(x='Class', y='Amount', data=df_clean)
# plt.title("Amount vs Class")
# plt.show()

corr = df_clean.corr()
# class_corr = corr['Class'].sort_values(ascending=False)
# print("\n\nCorrelation:")
# print(class_corr)

# plt.figure(figsize=(10,10))
# sns.heatmap(class_corr.to_frame())
# plt.title("Correlation Heatmap")
# plt.show()

important_features = corr['Class'][abs(corr['Class']) > 0.15]
# print("\n\nImportant features (|corr| > 0.15):")
# print(important_features)

# plt.figure()
# sns.heatmap(important_features.to_frame())
# plt.title("Important Features Heatmap")
# plt.show()

features = df_clean.drop('Class', axis=1)
target = df_clean['Class']

# scaler = StandardScaler()
# features['Amount'] = scaler.fit_transform(features['Amount'].values.reshape(-1,1))
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.2, random_state=1
)

# important_features_columns = corr['Class'][abs(corr['Class']) > 0.15] \
#     .drop('Class', errors='ignore') \
#     .index.tolist()

# smote = SMOTE(random_state=1)
# features_train_smote, target_train_smote = smote.fit_resample(features_train, target_train)

# print("\n\nTarget Train values Class Distribution before SMOTE:")
# print(target_train.value_counts())

# print("\n\nTarget Train values Class Distribution after SMOTE:")
# print(target_train_smote.value_counts())

# ------------------------------------------------------
# model = LogisticRegression()
# model.fit(features_train_smote, target_train_smote)
# target_predict = model.predict(features_test)

# Classification Report of Logistic Regression when trained using SMOTE data
#               precision    recall  f1-score   support
#
#            0       1.00      0.97      0.99     55031
#            1       0.06      0.91      0.12       102
#
#     accuracy                           0.97     55133
#    macro avg       0.53      0.94      0.55     55133
# weighted avg       1.00      0.97      0.99     55133
# -------------------------------------------------------

# ----------------------------------------------------------------
# model = LogisticRegression()
# model.fit(features_train[important_features_columns],target_train)
# target_predict = model.predict(features_test[important_features_columns])

# Classification Report  of Logistic Regression when trained using important features only
#               precision    recall  f1-score   support
#
#            0       1.00      1.00      1.00     55031
#            1       0.80      0.58      0.67       102
#
#     accuracy                           1.00     55133
#    macro avg       0.90      0.79      0.83     55133
# weighted avg       1.00      1.00      1.00     55133
# -------------------------------------------------------------------

# -----------------------------------------------------
# model = LogisticRegression()
# model.fit(features_train, target_train)
# target_predict = model.predict(features_test)

# Classification Report of Logistic Regression when trained using original data (Best among these three)
#               precision    recall  f1-score   support
#
#            0       1.00      1.00      1.00     55031
#            1       0.81      0.62      0.70       102
#
#     accuracy                           1.00     55133
#    macro avg       0.90      0.81      0.85     55133
# weighted avg       1.00      1.00      1.00     55133
# ------------------------------------------------------

# -------------------------------------------------------------
# model = RandomForestClassifier(n_estimators=50, random_state=42)
# model.fit(features_train, target_train)
# target_predict = model.predict(features_test)

# Classification Report of Random Forest when trained using original data
#               precision    recall  f1-score   support
#
#            0       1.00      1.00      1.00     55031
#            1       0.90      0.72      0.80       102
#
#     accuracy                           1.00     55133
#    macro avg       0.95      0.86      0.90     55133
# weighted avg       1.00      1.00      1.00     55133
# ---------------------------------------------------------------

# print("Classification Report")
# print(classification_report(target_test, target_predict))