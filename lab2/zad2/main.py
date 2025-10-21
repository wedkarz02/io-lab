import pandas as pd
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_data(path="./data/iris.csv"):
    df = pd.read_csv(path)
    numeric_cols = df.select_dtypes(include="number").columns
    X = df[numeric_cols]
    y = df["variety"] if "variety" in df.columns else None
    return X, y


def standardize_data(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)


def perform_pca(X_scaled):
    pca = PCA().fit(X_scaled)
    explained_var = pca.explained_variance_ratio_
    cum_var = explained_var.cumsum()
    print("\nWariancje poszczególnych składowych:", explained_var)
    print("Wariancje skumulowane:", cum_var)
    return explained_var, cum_var


def compress_data(X_scaled, y, n_components=2):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_components)])
    if y is not None:
        df_pca["variety"] = y
    print(
        f"\nZachowana wariancja dla {n_components} składowych:",
        sum(pca.explained_variance_ratio_),
    )
    return df_pca, pca


def plot_2d(df_pca):
    plt.figure(figsize=(8, 6))
    for variety, group in df_pca.groupby("variety"):
        plt.scatter(group.PC1, group.PC2, label=variety)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA – wizualizacja danych Iris w 2D")
    plt.legend()
    plt.show()


def plot_3d(df_pca):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for variety, group in df_pca.groupby("variety"):
        ax.scatter(group.PC1, group.PC2, group.PC3, label=variety)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("PCA – wizualizacja danych Iris w 3D")
    plt.legend()
    plt.show()


def main():
    print("Wczytywanie danych...")
    X, y = load_data()
    print("Standaryzacja danych...")
    X_scaled = standardize_data(X)
    print("Analiza PCA...")
    explained_var, cum_var = perform_pca(X_scaled)

    if cum_var[1] >= 0.95:
        n_components = 2
    elif cum_var[2] >= 0.95:
        n_components = 3
    else:
        n_components = 4

    print(f"\nZachowujemy {n_components} składowe (≥95% wariancji)")
    df_pca, pca = compress_data(X_scaled, y, n_components)

    if n_components == 2:
        plot_2d(df_pca)
    elif n_components == 3:
        plot_3d(df_pca)
    else:
        print("Nie można wizualizować więcej niż 3 wymiarów.")


if __name__ == "__main__":
    main()
