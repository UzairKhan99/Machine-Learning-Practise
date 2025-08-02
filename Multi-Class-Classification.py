import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score

file_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/GkDzb7bWrtvGXdPOfk6CIg/Obesity-level-prediction-dataset.csv"
data = pd.read_csv(file_path)

# Visualize the target variable 'NObeyesdad'
plt.figure(figsize=(10, 6))
sns.countplot(x="NObeyesdad", data=data)
plt.title("Distribution of Obesity Levels")
plt.xlabel("Obesity Level")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()


print(data.isnull().sum())

# Dataset summary
print(data.info())
print(data.describe())

num_cols = data.select_dtypes(include="float64").columns
scaler = StandardScaler()
data[num_cols] = scaler.fit_transform(data[num_cols])

cat_cols = data.select_dtypes(include="object").columns.tolist()
cat_cols.remove("NObeyesdad")

# 2. Apply one-hot encoding
ohe = OneHotEncoder(drop="first", sparse_output=False)
encoded = ohe.fit_transform(data[cat_cols])

# 3. Convert to DataFrame
encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cat_cols))

# 4. Combine with other columns
prepped_data = pd.concat([data.drop(cat_cols, axis=1), encoded_df], axis=1)


prepped_data["NObeyesdad"] = prepped_data["NObeyesdad"].astype("category").cat.codes

X = prepped_data.drop("NObeyesdad", axis=1)
y = prepped_data["NObeyesdad"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


model_ova = LogisticRegression(multi_class="ovr", max_iter=1000)

model_ova.fit(X_train, y_train)

y_pred_ova = model_ova.predict(X_test)

# Evaluation metrics for OvA
print("One-vs-All (OvA) Strategy")
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ova),2)}%")
