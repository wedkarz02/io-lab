import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt


def split_dataset(file_name, seed):
    df = pd.read_csv(file_name)
    X = df.iloc[:, :4].values
    y = df.iloc[:, 4].values
    return train_test_split(X, y, train_size=0.7, random_state=seed, stratify=y)


class Zad1:
    @staticmethod
    def classify_iris(row):
        # sl, sw, pl, pw = row
        _, _, pl, pw = row

        if pw < 0.5:
            return "setosa"
        elif pl > 4:
            return "virginica"
        else:
            return "versicolor"

    @staticmethod
    def main():
        _, test_x, _, test_y = split_dataset("data/iris_big.csv", 288568)

        length = len(test_y)
        good_pred = 0

        for i in range(length):
            if Zad1.classify_iris(test_x[i]) == test_y[i]:
                good_pred += 1

        print(good_pred)
        print(good_pred / length * 100, "%")


class Zad2:
    @staticmethod
    def main():
        x_train, x_test, y_train, y_test = split_dataset("data/iris_big.csv", 69420)

        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(x_train, y_train)

        feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

        plt.figure(figsize=(30, 20), dpi=300)
        plot_tree(
            clf,
            filled=True,
            feature_names=feature_names,
            class_names=clf.classes_,
            rounded=True,
            fontsize=10,
        )
        plt.title("decision tree", fontsize=20)
        plt.savefig("decision_tree.png", bbox_inches="tight")

        y_pred = clf.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"acc: {acc}")

        cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
        print("\nconfusion matrix:")
        print(pd.DataFrame(cm, index=clf.classes_, columns=clf.classes_))


class Zad3:
    @staticmethod
    def main():
        x_train, x_test, y_train, y_test = split_dataset("data/iris_big.csv", 12345)
        classifiers = {
            "3-NN": KNeighborsClassifier(n_neighbors=3),
            "5-NN": KNeighborsClassifier(n_neighbors=5),
            "11-NN": KNeighborsClassifier(n_neighbors=11),
            "Naive Bayes": GaussianNB(),
        }

        results = {}

        for name, clf in classifiers.items():
            print(f"\n--- {name} ---")
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = accuracy
            cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)

            print(f"acc: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print("confusion matrix:")
            print(pd.DataFrame(cm, index=clf.classes_, columns=clf.classes_))

        print("\n")
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        for i, (name, accuracy) in enumerate(sorted_results, 1):
            print(f"{i}. {name}: {accuracy:.4f} ({accuracy*100:.2f}%)")


if __name__ == "__main__":
    match sys.argv[1]:
        case "1":
            Zad1.main()
        case "2":
            Zad2.main()
        case "3":
            Zad3.main()
        case _:
            print("invalid arg")
