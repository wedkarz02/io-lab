import pandas as pd
from sklearn.model_selection import train_test_split


def in_range(num, low, high):
    return low <= num <= high


def split_dataset(file_name):
    df = pd.read_csv(file_name)
    return train_test_split(df.values, train_size=0.7, random_state=288568)


def classify_iris(row):
    # sl, sw, pl, pw = row
    _, _, pl, _ = row

    if in_range(pl, 1.2, 1.8):
        return "setosa"
    elif in_range(pl, 4.1, 4.9):
        return "virginica"
    else:
        return "versicolor"


def main():
    _, test_set = split_dataset("data/iris_big.csv")

    length = len(test_set)
    good_pred = 0

    for i in range(length):
        if classify_iris(test_set[i][:4]) == test_set[i][4]:
            good_pred += 1

    print(good_pred)
    print(good_pred / length * 100, "%")


if __name__ == "__main__":
    # df = pd.read_csv("data/iris_big.csv")
    # print(df.groupby("target_name").sample(3))
    main()
