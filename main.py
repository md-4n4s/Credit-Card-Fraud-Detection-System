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
from sklearn.metrics import confusion_matrix

pd.set_option('display.width', 0)

# Dataset exploration section
def dataset_overview(dataset):
    print("\n\nShape of the dataframe: ", dataset.shape)

    print("\n\nData types of the dataframe:")
    print(dataset.dtypes)

    print("\n\nFirst 5 rows of the dataframe:")
    print(dataset.head())

    print("\n\nNull values of the dataframe:")
    print(dataset.isnull().sum())

def drop_columns(dataset, columns):
    dataset = dataset.drop(columns, axis=1)
    return dataset

def remove_duplicates(dataset):
    print("\n\nShape of the dataframe before removing duplicates: ", dataset.shape)

    df_clean = dataset.drop_duplicates()

    print("\n\nShape of the dataframe after removing duplicates: ", df_clean.shape)

    return df_clean

# Analysis & visualization
def dataset_analysis(dataset):
    print("\n\nSummary Statistics:")
    print(dataset.describe().T)

    print("\n\nFraud vs Non-Fraud cases:")
    print(dataset['Class'].value_counts())

    print("\n\nFraud vs Non-Fraud comparison:")
    print(dataset.groupby('Class').mean().T)

def plot_class_analysis(dataset):
    sns.countplot(x='Class', data=dataset)
    plt.title("Fraud vs Non-Fraud")
    plt.show()

    sns.boxplot(x='Class', y='Amount', data=dataset)
    plt.title("Amount vs Class")
    plt.show()

def correlation_analysis(dataset):
    corr = dataset.corr()
    class_corr = corr['Class'].sort_values(ascending=False)

    print("\n\nCorrelation:")
    print(class_corr)

    plt.figure(figsize=(10, 10))
    sns.heatmap(class_corr.to_frame())
    plt.title("Correlation Heatmap")
    plt.show()

def get_important_features(dataset, threshold=0.15):
    corr = dataset.corr()

    important_features = corr['Class'][abs(corr['Class']) > threshold]
    print(f"\n\nImportant features (|corr| > {threshold}):")
    print(important_features)

    important_features_columns = corr['Class'][abs(corr['Class']) > threshold] \
        .drop('Class', errors='ignore') \
        .index.tolist()

    print("\n\nImportant feature columns list:")
    print(important_features_columns)

    return important_features, important_features_columns

# Data preparation
def split_features_target(dataset):
    features = dataset.drop('Class', axis=1)
    target = dataset['Class']

    return features, target

def scale_and_split(features, target, scale_cols):
    scaler = StandardScaler()

    features[scale_cols] = scaler.fit_transform(features[scale_cols])

    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=0.2, random_state=1
    )

    return features_train, features_test, target_train, target_test, scaler

def apply_smote(features_train, target_train, random_state=1):
    smote = SMOTE(random_state=random_state)

    features_train_smote, target_train_smote = smote.fit_resample(
        features_train, target_train
    )

    print("\n\nClass distribution in Training Data (Before SMOTE):")
    print(target_train.value_counts())

    print("\n\nClass distribution in Training Data (After SMOTE):")
    print(target_train_smote.value_counts())

    return features_train_smote, target_train_smote

# Model training
def train_logistic_regression(features_train, target_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(features_train, target_train)
    return model

def train_random_forest(features_train, target_train, n_estimators=50, random_state=42):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )

    model.fit(features_train, target_train)
    return model

def predict(model, features_test):
    return model.predict(features_test)

# Evaluation
def evaluate_model(model_name, target_test, target_predict):
    print("\n\nMODEL:", model_name)

    print("\n\nClassification Report")
    print(classification_report(target_test, target_predict))

    print("\n\nConfusion Matrix")
    print(confusion_matrix(target_test, target_predict))

def predict_fraud(model, features):
    prediction = model.predict(features)
    probability = model.predict_proba(features)

    print("\nPrediction:", "FRAUD" if prediction[0] == 1 else "NORMAL")
    print("Confidence (Normal, Fraud):", probability[0])

    return prediction[0]

def get_fraud_metrics(model, features_test, target_test):
    target_predict = model.predict(features_test)

    report = classification_report(target_test, target_predict, output_dict=True)

    fraud_class = report['1']

    return fraud_class['precision'], fraud_class['recall'], fraud_class['f1-score']

# Model comparison
def compare_models(models, names, features_tests, target_test):
    precisions = []
    recalls = []
    f1s = []

    for model, feature_test in zip(models, features_tests):
        p, r, f = get_fraud_metrics(model, feature_test, target_test)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    x = range(len(names))

    plt.figure(figsize=(10, 5))

    plt.bar(x, precisions, width=0.25, label="Precision")
    plt.bar([i + 0.25 for i in x], recalls, width=0.25, label="Recall")
    plt.bar([i + 0.50 for i in x], f1s, width=0.25, label="F1-score")

    plt.xticks([i + 0.25 for i in x], names)
    plt.title("Model Comparison (Fraud Class)")
    plt.legend()
    plt.show()

# Main pipeline
def main():
    # Load dataset
    df = pd.read_csv('creditcard.csv')

    # Basic inspection
    dataset_overview(df)

    # Clean data
    df_clean = remove_duplicates(df)

    # Analysis
    dataset_analysis(df_clean)
    plot_class_analysis(df_clean)
    correlation_analysis(df_clean)

    # Feature selection
    important_features, important_features_columns = get_important_features(df_clean)

    # Split data
    features, target = split_features_target(df_clean)

    features_train, features_test, target_train, target_test, scaler = scale_and_split(
        features, target, scale_cols=features.columns
    )

    # Handle imbalance
    features_train_smote, target_train_smote = apply_smote(features_train, target_train)

    # Logistic Regression models
    model1 = train_logistic_regression(features_train_smote, target_train_smote)
    predict1 = predict(model1, features_test)
    evaluate_model("Logistic Regression (SMOTE)",target_test, predict1)

    model2 = train_logistic_regression(
        features_train[important_features_columns],
        target_train
    )
    predict2 = predict(model2, features_test[important_features_columns])
    evaluate_model("Logistic Regression (Important Features)", target_test, predict2)

    model3 = train_logistic_regression(features_train, target_train)
    predict3 = predict(model3, features_test)
    evaluate_model("Logistic Regression (Original Data)", target_test, predict3)

    # Random Forest models
    model4 = train_random_forest(features_train_smote, target_train_smote)
    predict4 = predict(model4, features_test)
    evaluate_model("Random Forest (SMOTE)", target_test, predict4)

    model5 = train_random_forest(
        features_train[important_features_columns],
        target_train
    )
    predict5 = predict(model5, features_test[important_features_columns])
    evaluate_model("Random Forest (Important Features)", target_test, predict5)

    model6 = train_random_forest(features_train, target_train)
    predict6 = predict(model6, features_test)
    evaluate_model("Random Forest (Original Data)", target_test, predict6)

    # Single prediction example
    sample = features_test.iloc[0:1]
    predict_fraud(model4, sample)

    # Model comparison
    models = [model1, model2, model3, model4, model5, model6]

    names = [
        "SMOTE LR",
        "Important Features LR",
        "Original LR",
        "SMOTE RF",
        "Important Features RF",
        "Original RF"
    ]

    feature_tests = [
        features_test,
        features_test[important_features_columns],
        features_test,

        features_test,
        features_test[important_features_columns],
        features_test
    ]

    compare_models(models, names, feature_tests, target_test)

if __name__ == "__main__":
    main()