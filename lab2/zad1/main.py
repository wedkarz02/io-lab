import pandas as pd


def load_data(filename):
    return pd.read_csv(filename)


def analyze_missing_data(df):
    print("\na) BRAKUJĄCE DANE:")
    missing_counts = df.isnull().sum()
    print("Liczba brakujących wartości w każdej kolumnie:")
    print(missing_counts)
    print(f"Łączna liczba brakujących wartości: {missing_counts.sum()}")

    print("\nSTATYSTYKI BAZY PRZED NAPRAWĄ:")
    print(df.describe(include="all"))

    return missing_counts


def fix_numeric_data(df, numeric_columns):
    print("\nb) NAPRAWA DANYCH NUMERYCZNYCH:")

    total_fixed = 0
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

        out_of_range_mask = (df[col] <= 0) | (df[col] >= 15) | (df[col].isnull())
        out_of_range_count = out_of_range_mask.sum()

        if out_of_range_count > 0:
            median_value = df[col].median()
            df.loc[out_of_range_mask, col] = median_value
            print(
                f"Kolumna {col}: naprawiono {out_of_range_count} wartości (mediana: {median_value:.2f})"
            )
            total_fixed += out_of_range_count

    print(f"Łącznie naprawiono {total_fixed} wartości numerycznych")
    return df


def correct_species_name(species, df, correct_species):
    if pd.isna(species):
        return "Setosa"

    species = str(species).strip()

    corrections = {
        "setosa": "Setosa",
        "SETOSA": "Setosa",
        "versicolor": "Versicolor",
        "VERSICOLOR": "Versicolor",
        "virginica": "Virginica",
        "VIRGINICA": "Virginica",
        "Virginica": "Virginica",
        "Versicolour": "Versicolor",
    }

    if species in corrections:
        return corrections[species]
    elif species in correct_species:
        return species
    else:
        return df["variety"].mode()[0] if not df["variety"].mode().empty else "Setosa"


def fix_species_data(df):
    print("\nc) NAPRAWA GATUNKÓW:")

    correct_species = ["Setosa", "Versicolor", "Virginica"]

    print("Unikalne wartości przed naprawą:")
    print(df["variety"].unique())

    initial_species = df["variety"].copy()

    df["variety"] = df["variety"].apply(
        lambda x: correct_species_name(x, df, correct_species)
    )

    changed_species = initial_species != df["variety"]
    changes_count = changed_species.sum()

    if changes_count > 0:
        print(f"Naprawiono {changes_count} niepoprawnych wpisów gatunków")
        for idx in df[changed_species].index:
            print(f"  Wiersz {idx}: '{initial_species[idx]}' -> '{df['variety'][idx]}'")

    print("Unikalne wartości po naprawie:")
    print(df["variety"].unique())

    return df, changes_count


def display_final_statistics(df, numeric_columns):
    print("\n=== STATYSTYKI PO NAPRAWIE ===")
    print("Liczba brakujących wartości:")
    print(df.isnull().sum())
    print(f"Łączna liczba brakujących wartości: {df.isnull().sum().sum()}")

    print("\nStatystyki numeryczne:")
    print(df[numeric_columns].describe())

    print("\nRozkład gatunków:")
    print(df["variety"].value_counts())


def verify_ranges(df, numeric_columns):
    print("\n=== WERYFIKACJA ZAKRESÓW ===")
    all_ok = True
    for col in numeric_columns:
        min_val = df[col].min()
        max_val = df[col].max()
        is_ok = min_val > 0 and max_val < 15
        status = "OK" if is_ok else "BŁĄD"
        print(f"{col}: zakres [{min_val:.2f}, {max_val:.2f}] - {status}")

        if not is_ok:
            all_ok = False

    return all_ok


def save_corrected_data(df, output_filename):
    df.to_csv(output_filename, index=False)
    print(f"\nPoprawiona baza danych zapisana jako '{output_filename}'")


def main():
    print("=== ANALIZA I NAPRAWA BAZY DANYCH IRIS ===")

    numeric_columns = ["sepal.length", "sepal.width", "petal.length", "petal.width"]

    df = load_data("data/iris_with_errors.csv")

    missing_stats = analyze_missing_data(df)

    df = fix_numeric_data(df, numeric_columns)

    df, species_changes = fix_species_data(df)

    display_final_statistics(df, numeric_columns)

    ranges_ok = verify_ranges(df, numeric_columns)

    save_corrected_data(df, "data/iris_corrected.csv")

    print("\n=== PODSUMOWANIE ===")
    print(f"Naprawiono {missing_stats.sum()} brakujących wartości")
    print(f"Naprawiono {species_changes} błędnych gatunków")
    print(
        f"Wszystkie dane numeryczne w zakresie (0;15): {'TAK' if ranges_ok else 'NIE'}"
    )

    return df


if __name__ == "__main__":
    df_corrected = main()
