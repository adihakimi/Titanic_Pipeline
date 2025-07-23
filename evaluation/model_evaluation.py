import joblib
import pandas as pd
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse



def evaluate(model_path, test_data_path, output_dir):
    model = joblib.load(model_path)
    test_df = pd.read_csv(test_data_path)
    X_test = test_df.drop(columns=["Survived"])
    y_test = test_df["Survived"]
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    conf_matrix = confusion_matrix(y_test, preds)
    print(f"Test Accuracy: {acc:.4f}")

    # Save evaluation results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'evaluation_report.txt'), 'w') as f:
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(conf_matrix))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='/opt/ml/input/data/output')
    parser.add_argument('--test-data', type=str, default='/opt/ml/input/data/test')
    parser.add_argument('--model-dir', type=str, default='/opt/ml/model') 
    args = parser.parse_args()
    evaluate(args.model_dir, args.test_data, args.output_dir)




