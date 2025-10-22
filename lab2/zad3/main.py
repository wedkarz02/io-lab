import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

df = pd.read_csv("./data/iris.csv")

features = ["sepal.length", "sepal.width"]
X = df[features]
y = df["variety"]

print("Statystyki dla danych oryginalnych:")
print(X.describe())

scaler_z = StandardScaler()
X_z = scaler_z.fit_transform(X)

scaler_mm = MinMaxScaler()
X_mm = scaler_mm.fit_transform(X)


def plot_data(X_plot, title, filename):
    plt.figure(figsize=(6, 5))
    for variety, group in (
        pd.DataFrame(X_plot, columns=features).assign(variety=y).groupby("variety")
    ):
        plt.scatter(group["sepal.length"], group["sepal.width"], label=variety)
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Sepal Width (cm)")
    plt.title(title)
    plt.legend()
    plt.savefig(filename)


plot_data(X.values, "Original Dataset", "original.png")
plot_data(X_z, "Z-Core Scaled Dataset", "zcore.png")
plot_data(X_mm, "Min-Max Normalised Dataset", "minmax.png")

print("\nZ-Score (standaryzowane): mean =", X_z.mean(axis=0), " std =", X_z.std(axis=0))
print("Min-Max (0-1): min =", X_mm.min(axis=0), " max =", X_mm.max(axis=0))
