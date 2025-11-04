import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix


class Zad1:
    @staticmethod
    def f(x):
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def forward_pass(wiek, waga, wzrost):
        neuron1 = Zad1.f(wiek * -0.46122 + waga * 0.97314 - wzrost * 0.39203 + 0.80109)
        neuron2 = Zad1.f(wiek * 0.78548 + waga * 2.10584 - wzrost * 0.57847 + 0.43529)
        return neuron1 * -0.81546 + neuron2 * 1.03775 - 0.2368

    @staticmethod
    def main():
        print(Zad1.forward_pass(25, 67, 180))
        print(Zad1.forward_pass(48, 97, 178))


class Zad2:
    @staticmethod
    def split_dataset(file_name, seed):
        df = pd.read_csv(file_name)
        df["target_name"] = df["target_name"].replace(
            ["setosa", "versicolor", "virginica"], [0, 1, 2]
        )

        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        return train_test_split(x, y, train_size=0.7, random_state=seed)

    @staticmethod
    def normalize(x_train, x_test):
        scaler = StandardScaler()
        return scaler.fit_transform(x_train), scaler.transform(x_test)

    @staticmethod
    def main():
        x_train, x_test, y_train, y_test = Zad2.split_dataset(
            "./data/iris_big.csv", 69420
        )
        x_train, x_test = Zad2.normalize(x_train, x_test)

        topologies = {
            "4-2-1": (2,),
            "4-3-1": (3,),
            "4-3-3-1": (3, 3),
        }

        for name, top in topologies.items():
            model = MLPClassifier(
                hidden_layer_sizes=top, max_iter=2000, random_state=69420
            )
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            acc = accuracy_score(y_test, predictions)
            cm = confusion_matrix(y_test, predictions)
            print(f"\n======== {name} topology ========")
            print(f"acc: {acc}")
            print(f"confusion matrix:\n{cm}")


class Zad3(Zad2):
    @staticmethod
    def split_dataset(file_name, seed):
        df = pd.read_csv(file_name)
        df["target_name"] = df["target_name"].replace(
            ["setosa", "versicolor", "virginica"], [0, 1, 2]
        )

        s = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        y_onehot = np.zeros((y.size, 3))  # type: ignore
        y_onehot[np.arange(y.size), y] = 1  # type: ignore
        return train_test_split(s, y_onehot, train_size=0.7, random_state=seed)

    @staticmethod
    def main():
        x_train, x_test, y_train, y_test = Zad3.split_dataset(
            "./data/iris_big.csv", 69420
        )
        x_train, x_test = Zad3.normalize(x_train, x_test)

        topologies = {
            "4-2-3": (2,),
            "4-3-3": (3,),
            "4-3-3-3": (3, 3),
        }

        for name, top in topologies.items():
            model = MLPClassifier(
                hidden_layer_sizes=top,
                activation="relu",
                max_iter=2000,
                random_state=69420,
            )
            model.fit(x_train, y_train)
            preds = model.predict(x_test)

            y_true = np.argmax(y_test, axis=1)
            y_pred = np.argmax(preds, axis=1)  # type: ignore

            acc = accuracy_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred)
            print(f"\n======== {name} topology ========")
            print(f"acc: {acc}")
            print(f"confusion matrix:\n{cm}")


class Zad4:
    @staticmethod
    def split_dataset(file_name, seed):
        df = pd.read_csv(file_name)
        df.columns = [c.strip() for c in df.columns]

        if "class" not in df.columns:
            raise ValueError("Plik nie zawiera kolumny 'class' z etykietami.")
        df["class_binary"] = df["class"].map(
            {"tested_negative": 0, "tested_positive": 1}
        )
        if df["class_binary"].isna().any():

            df["class_binary"] = LabelEncoder().fit_transform(df["class"].astype(str))

        zero_as_na_candidates = [
            "glucose-concentr",
            "blood-pressure",
            "skin-thickness",
            "insulin",
            "mass-index",
            "glucose",
            "bloodpressure",
            "bmi",
        ]
        cols_zero_as_na = [c for c in zero_as_na_candidates if c in df.columns]

        for c in cols_zero_as_na:
            df[c] = df[c].replace(0, np.nan)

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        num_cols = [c for c in num_cols if c != "class_binary"]
        cols_with_nan = [c for c in num_cols if df[c].isna().sum() > 0]

        if cols_with_nan:
            imputer = SimpleImputer(strategy="median")
            df[cols_with_nan] = imputer.fit_transform(df[cols_with_nan])

        feature_cols = [c for c in df.columns if c not in ["class", "class_binary"]]

        X = df[feature_cols].select_dtypes(include=[np.number]).values
        y = df["class_binary"].astype(int).values

        return train_test_split(X, y, train_size=0.7, random_state=seed, stratify=y)

    @staticmethod
    def normalize(x_train, x_test):
        scaler = StandardScaler()
        return scaler.fit_transform(x_train), scaler.transform(x_test)

    @staticmethod
    def main():
        seed = 69420
        x_train, x_test, y_train, y_test = Zad4.split_dataset(
            "./data/diabetes.csv", seed
        )
        x_train, x_test = Zad4.normalize(x_train, x_test)

        classifiers = []
        classifiers.append(("DecisionTree", DecisionTreeClassifier(random_state=seed)))
        classifiers.append(("kNN_k5", KNeighborsClassifier(n_neighbors=5)))
        classifiers.append(("GaussianNB", GaussianNB()))
        mlp_topologies = {
            "MLP_50_relu": {"hidden_layer_sizes": (50,), "activation": "relu"},
            "MLP_100_50_relu": {"hidden_layer_sizes": (100, 50), "activation": "relu"},
            "MLP_50_tanh": {"hidden_layer_sizes": (50,), "activation": "tanh"},
            "MLP_50_logistic": {"hidden_layer_sizes": (50,), "activation": "logistic"},
        }
        for name, cfg in mlp_topologies.items():
            classifiers.append(
                (
                    name,
                    MLPClassifier(
                        hidden_layer_sizes=cfg["hidden_layer_sizes"],
                        activation=cfg["activation"],
                        max_iter=5000,
                        random_state=seed,
                    ),
                )
            )

        summary = []
        for name, clf in classifiers:
            clf.fit(x_train, y_train)
            preds = clf.predict(x_test)
            acc = accuracy_score(y_test, preds)
            cm = confusion_matrix(y_test, preds)

            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                tn = cm[0, 0] if cm.shape[0] > 0 and cm.shape[1] > 0 else 0
                fp = cm[0, 1] if cm.shape[0] > 0 and cm.shape[1] > 1 else 0
                fn = cm[1, 0] if cm.shape[0] > 1 and cm.shape[1] > 0 else 0
                tp = cm[1, 1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0

            print(f"\n======== {name} ========")
            print(f"acc: {acc:.4f}")
            print("confusion matrix:")
            print(cm)
            summary.append(
                {
                    "name": name,
                    "accuracy": acc,
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp),
                }
            )

        summary_df = (
            pd.DataFrame(summary)
            .sort_values(by="accuracy", ascending=False)
            .reset_index(drop=True)
        )
        plt.figure(figsize=(10, 6))
        plt.bar(summary_df["name"], summary_df["accuracy"])
        plt.ylim(0, 1)
        plt.ylabel("Dokładność")
        plt.xlabel("Klasyfikator")
        plt.title("Dokładności klasyfikatorów (Zad4)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig("zad4_accuracies.png")
        print("\nWykres dokładności zapisany jako 'zad4_accuracies.png'")

        summary_df["fn_rate"] = summary_df.apply(
            lambda r: (
                r["fn"] / (r["fn"] + r["tp"]) if (r["fn"] + r["tp"]) > 0 else np.nan
            ),
            axis=1,
        )
        summary_df["fp_rate"] = summary_df.apply(
            lambda r: (
                r["fp"] / (r["fp"] + r["tn"]) if (r["fp"] + r["tn"]) > 0 else np.nan
            ),
            axis=1,
        )

        print("\n--- FP / FN dla modeli ---")
        for _, row in summary_df.iterrows():
            print(
                f"{row['name']}: FP={row['fp']} (rate={row['fp_rate']:.3f}), FN={row['fn']} (rate={row['fn_rate']:.3f}), acc={row['accuracy']:.3f}"
            )

        min_fn_idx = summary_df["fn"].idxmin()
        min_fn_row = summary_df.loc[min_fn_idx]
        print("\nModel minimalizujący FN (najmniej przeoczonych chorych):")
        print(min_fn_row[["name", "fn", "fn_rate", "accuracy"]])

        summary_df.to_csv("zad4_summary.csv", index=False)
        print("\nPodsumowanie zapisane jako 'zad4_summary.csv'")


if __name__ == "__main__":
    match sys.argv[1]:
        case "1":
            Zad1.main()
        case "2":
            Zad2.main()
        case "3":
            Zad3.main()
        case "4":
            Zad4.main()
        case _:
            print("invalid arg")
