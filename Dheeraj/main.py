import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load your datasetspp
data1 = pd.read_csv(
    r"C:\Users\tudim\OneDrive\Desktop\mini_project\dheerajmini\Dheeraj\collegePlace.csv")
data2 = pd.read_csv(
    r"C:\Users\tudim\OneDrive\Desktop\mini_project\dheerajmini\Dheeraj\Train_Data.csv")

col = ["gender", "degree_t", "workex", "ssc_b",
       'hsc_b', 'specialisation', "hsc_s"]
mat = data2.drop(["mba_p", "status"], axis=1)
label_encoders = {}

for c in col:
    le = LabelEncoder()  # Initialize LabelEncoder for each column
    mat[c] = le.fit_transform(mat[c])
    label_encoders[c] = le

# Define feature (X) and target (y)
y_mat = le.fit_transform(data2["status"])

# Create and train a Random Forest Regressor model
model = RandomForestClassifier(criterion="log_loss", n_estimators=300)
model.fit(mat, y_mat)


with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
