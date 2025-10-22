import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset():
    df = pd.read_csv("data/iris_big.csv")
    return train_test_split(df.values, train_size=0.7, random_state=288568)


def classify_iris(sl, sw, pl, pw):
    if sl > 4:
        return "setosa"
    elif pl <= 5:
        return "virginica"
    else:
        return "versicolor"


def main():
    (train_set, test_set) = split_dataset()
    good_pred = 0
    lenght = test_set.shape[0]

    for i in range(lenght):
        if (
            classify_iris(
                test_set[i][0], test_set[i][1], test_set[i][2], test_set[i][3]
            )
            == test_set[i][4]
        ):
            good_pred += 1

    print(good_pred)
    print(good_pred / test_set.shape[0] * 100, "%")


if __name__ == "__main__":
    main()
