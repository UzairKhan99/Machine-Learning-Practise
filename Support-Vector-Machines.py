# Import required libraries
from __future__ import print_function
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC

# Load dataset
url = "drug100.csv"
raw_data = pd.read_csv(url)

# Prepare features and target
X = StandardScaler().fit_transform(raw_data.iloc[:, 1:30]).values
y = raw_data.iloc[:, 30].values

# Normalize data and split into train/test sets
X = normalize(X, norm="l1")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Compute sample weights for class imbalance
w_train = compute_sample_weight("balanced", y_train)

# Initialize and train Decision Tree
dt = DecisionTreeClassifier(max_depth=4, random_state=35)
dt.fit(X_train, y_train, sample_weight=w_train)

# Initialize and train SVM
svm = LinearSVC(
    class_weight="balanced", random_state=31, loss="hinge", fit_intercept=False
)
svm.fit(X_train, y_train)

# Evaluate models
y_pred_dt = dt.predict_proba(X_test)[:, 1]
roc_auc_dt = roc_auc_score(y_test, y_pred_dt)
print("Decision Tree ROC-AUC score: {0:.3f}".format(roc_auc_dt))

y_pred_svm = svm.decision_function(X_test)
roc_auc_svm = roc_auc_score(y_test, y_pred_svm)
print("SVM ROC-AUC score: {0:.3f}".format(roc_auc_svm))
