import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics

DATA_URL = "drug200.csv"
drug_data = pd.read_csv(DATA_URL)

label_encoder = LabelEncoder()

drug_data["Sex"] = label_encoder.fit_transform(drug_data["Sex"])
drug_data["BP"] = label_encoder.fit_transform(drug_data["BP"])
drug_data["Cholesterol"] = label_encoder.fit_transform(drug_data["Cholesterol"])
drug_data["Drug"] = label_encoder.fit_transform(drug_data["Drug"])

print(drug_data.corr()["Drug"])
y = drug_data["Drug"]
X = drug_data.drop("Drug", axis=1)

X_trainset, X_testset, y_trainset, y_testset = train_test_split(
    X, y, test_size=0.3, random_state=32
)
drugTree = DecisionTreeClassifier(criterion="gini", max_depth=4)
drugTree.fit(X_trainset, y_trainset)
tree_predictions = drugTree.predict(X_testset)
print(
    "Decision Trees's Accuracy: ", metrics.accuracy_score(y_testset, tree_predictions)
)
