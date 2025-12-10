import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from app import global_data

def auto_convert_target(y):

    if y.dtype.kind in "fc":
        median = np.median(y)
        return (y > median).astype(int)

    return y


def handle_upload(file):
    if file.filename.endswith(".csv"):
        global_data.data = pd.read_csv(file.file)
    elif file.filename.endswith(".xlsx"):
        global_data.data = pd.read_excel(file.file)
    else:
        return {"error": "Invalid file format"}

    return {
        "rows": global_data.data.shape[0],
        "columns": global_data.data.shape[1],
        "column_names": global_data.data.columns.tolist(),
        "preview": global_data.data.head(5).to_dict(orient="records")
    }

def handle_preprocess(standardize, normalize):
    data = global_data.data

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    if standardize:
        X = StandardScaler().fit_transform(X)

    if normalize:
        X = MinMaxScaler().fit_transform(X)

    global_data.data = pd.DataFrame(X)
    global_data.data["target"] = y

    return {"status": "Preprocessing applied successfully"}

def handle_split(split_ratio):
    X = global_data.data.iloc[:, :-1]
    y = global_data.data.iloc[:, -1]

    y = auto_convert_target(y)

    test_size = 1 - (split_ratio / 100)

    if len(X) * (1 - test_size) < 1:
        return {
            "error": "Train set would be empty. Increase train percentage."
        }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    global_data.X_train = X_train
    global_data.X_test = X_test
    global_data.y_train = y_train
    global_data.y_test = y_test

    return {
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "converted_target": "yes" if y.dtype.kind not in "iu" else "no"
    }


def handle_train(model):
    y_unique = global_data.y_train.nunique()

    if y_unique < 2:
        return {
            "error": "Only one class present in training data. Choose a different dataset or change split."
        }

    if model == "logistic":
        clf = LogisticRegression()
    else:
        clf = DecisionTreeClassifier()

    clf.fit(global_data.X_train, global_data.y_train)
    preds = clf.predict(global_data.X_test)

    acc = accuracy_score(global_data.y_test, preds)
    precision = precision_score(global_data.y_test, preds, zero_division=0)
    recall = recall_score(global_data.y_test, preds, zero_division=0)
    f1 = f1_score(global_data.y_test, preds, zero_division=0)

    return {
        "accuracy": round(acc * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1 * 100, 2)
    }

