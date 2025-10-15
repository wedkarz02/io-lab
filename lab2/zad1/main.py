import numpy as np
import pandas as pd

def main(bfile, gfile):
    df = pd.read_csv(bfile)
    print(df.head)


if __name__ == "__main__":
    BROKEN_FILE = "data/iris_with_errors.csv"
    GOOD_FILE = "data/iris_with_errors.csv"
    main(BROKEN_FILE, GOOD_FILE)
