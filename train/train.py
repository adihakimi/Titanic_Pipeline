import argparse
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib


def train(train_data, test_data, model_path):
    # Load the dataset
    #if not os.path.exists(data_path):
    #    raise FileNotFoundError(f"Data file {data_path} does not exist.")
    
    train_df = pd.read_csv(train_data)
    test_df = pd.read_csv(test_data)

    X_train = train_df.drop(columns=["Survived"])
    y_train = train_df["Survived"]

    X_test = test_df.drop(columns=["Survived"])
    y_test = test_df["Survived"]

    print(X_test.isnull().sum())
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test,preds)
    print(f"Test Accuracy: {acc:.4f}")

    joblib.dump(model, os.path.join(model_path, 'titanic.joblib'))
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', type=str, default='/opt/ml/input/data/train')
    parser.add_argument('--test-data', type=str, default='/opt/ml/input/data/test')
    parser.add_argument('--model-dir', type=str, default='/opt/ml/model')
    args = parser.parse_args()
    train(args.train_data, args.test_data, args.model_dir)