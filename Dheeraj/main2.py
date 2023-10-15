import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle


# data_aquisition
data2 = pd.read_csv(
    r"C:\Users\tudim\OneDrive\Desktop\mini_project\dheerajmini\Dheeraj\Train_Data.csv")

# data preprocessing
col = ["gender", "degree_t", "workex", "ssc_b",
       'hsc_b', 'specialisation', "hsc_s"]
x_train = data2.drop(["mba_p", "status"], axis=1)
label_encoders = {}

for c in col:
    le = LabelEncoder()
    x_train[c] = le.fit_transform(x_train[c])
    label_encoders[c] = le

# training and testing data
x_train = x_train
y_train = le.fit_transform(data2["status"])

# Hyperparameters
model_params = {
    'logistic': {
        'model': LogisticRegression(),
        'params': {
            'fit_intercept': [True]
        }
    },

    'ridge': {
        'model': Ridge(),
        'params': {
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs']
        }
    },

    'lasso': {
        'model': Lasso(),
        'params': {
            'alpha': [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1],
            'selection': ['cyclic', 'random']
        }
    },
    'decision tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'max_leaf_nodes': [5, 6, 7, 8, 9, 10],
            'criterion': ['entropy'],
            'max_depth': [1, 2, 3]

        }
    },
    'RandomForest': {
        'model': RandomForestClassifier(),
        'params': {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'n_estimators': [100, 200, 300]
        }
    }

}

scores = []
for mName, mp in model_params.items():
    clf = GridSearchCV(mp['model'], mp['params'],
                       cv=5, return_train_score=False)
    clf.fit(x_train, y_train)
    scores.append({
        'model': mName,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
ff = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])

# choosing the best model
best_model = ff.sort_values('best_score', ascending=False).iloc[0]['model']
best_parameters = ff.sort_values(
    'best_score', ascending=False).iloc[0]['best_params']

if best_model == "RandomForest":
    model = RandomForestClassifier(
        criterion=best_parameters["criterion"], n_estimators=best_parameters["n_estimators"])
elif best_model == "logistic":
    model = LogisticRegression(fit_intercept=best_parameters["fit_intercept"])
elif best_model == "ridge":
    model = Ridge(solver=best_parameters["solver"])
elif best_model == "lasso":
    model = Lasso(alpha=best_parameters["alpha"],
                  selection=best_parameters["selection"])
elif best_model == "decision tree":
    model = DecisionTreeClassifier(
        max_leaf_nodes=best_parameters["max_leaf_nodes"],
        criterion=best_parameters["criterion"],
        max_depth=best_parameters["max_depth"]
    )
else:
    # Handle other models or provide a default model if best_model is not recognized
    model = None

# training the model on train data
model.fit(x_train, y_train)


# creating pickle file
with open('model1.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
