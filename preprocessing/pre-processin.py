import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET = "titanic.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
REMOVE_COLUMNS = ['PassengerId', 'Name', 'Ticket', 'Cabin']

def preprocessing(file_path, train_output, test_output):

    df = pd.read_csv(file_path)

    df.drop(columns=REMOVE_COLUMNS, inplace=True)

    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)


    df_train, df_test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    df_train.to_csv(f"{train_output}/train.csv", index=False)
    df_test.to_csv(f"{test_output}/train.csv", index=False)


    print(f"Training data saved to {train_output}")
    print(f"Testing data saved to {test_output}")
    print("Preprocessing completed successfully.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, default=DATASET, help="Path to the Titanic dataset CSV file")
    parser.add_argument("--train-output", type=str, default="/Users/adihakimi/train", help="Path to save the training dataset")
    parser.add_argument("--test-output", type=str, default="/Users/adihakimi/test", help="Path to save the testing dataset")
    args = parser.parse_args()

    preprocessing(args.input_data, args.train_output, args.test_output)