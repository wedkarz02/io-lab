import pandas as pd


def main():
    df = pd.read_csv("data/iris_big.csv")
    print(df)


if __name__ == "__main__":
    main()
