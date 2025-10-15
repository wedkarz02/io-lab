import math
import random
import matplotlib.pyplot as plt

def oblicz_zasieg(v0, h, kat_stopnie):
    kat_rad = math.radians(kat_stopnie)
    g = 9.81
    sin_alpha = math.sin(kat_rad)
    cos_alpha = math.cos(kat_rad)
    zasieg = (v0 * sin_alpha + math.sqrt(v0**2 * sin_alpha**2 + 2 * g * h)) * (v0 * cos_alpha) / g
    return zasieg

def rysuj_trajektorie(v0, h, kat_stopnie):
    kat_rad = math.radians(kat_stopnie)
    g = 9.81
    sin_alpha = math.sin(kat_rad)
    cos_alpha = math.cos(kat_rad)
    zasieg = (v0 * sin_alpha + math.sqrt(v0**2 * sin_alpha**2 + 2 * g * h)) * (v0 * cos_alpha) / g
    x_values = []
    y_values = []
    for x in range(0, int(zasieg) + 10, 5):
        y = - (g / (2 * v0**2 * cos_alpha**2)) * x**2 + (sin_alpha / cos_alpha) * x + h
        if y >= 0:
            x_values.append(x)
            y_values.append(y)
    plt.figure(figsize=(12, 6))
    plt.plot(x_values, y_values, 'b-', linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Odległość [m]', fontsize=12)
    plt.ylabel('Wysokość [m]', fontsize=12)
    plt.title(f'Trajektoria pocisku - kąt: {kat_stopnie}°', fontsize=14)
    plt.xlim(0, max(x_values) * 1.1)
    plt.ylim(0, max(y_values) * 1.1)
    plt.tight_layout()
    plt.savefig('trajektoria.png', dpi=150)

if __name__ == "__main__":
    v0 = 50
    h = 100
    cel_odleglosc = random.randint(50, 340)
    print(f"Cel znajduje się w odległości {cel_odleglosc} metrów.")
    print(f"Aby go trafić, pocisk musi upaść w zakresie [{cel_odleglosc-5}, {cel_odleglosc+5}] metrów.")
    print("Prędkość początkowa: 50 m/s, Wysokość: 100 m")
    print("-" * 50)
    liczba_prob = 0
    trafiony = False
    while not trafiony:
        try:
            kat = float(input("Podaj kąt strzału (w stopniach): "))
            liczba_prob += 1
            zasieg = oblicz_zasieg(v0, h, kat)
            print(f"Próba {liczba_prob}: Kąt {kat}° -> Pocisk upadł w {zasieg:.2f} metrów")
            if cel_odleglosc - 5 <= zasieg <= cel_odleglosc + 5:
                print("Cel trafiony!")
                print(f"Liczba prób: {liczba_prob}")
                trafiony = True
                rysuj_trajektorie(v0, h, kat)
                print("Wykres zapisany jako 'trajektoria.png'")
            else:
                if zasieg < cel_odleglosc - 5:
                    print("Pocisk nie doleciał do celu!")
                else:
                    print("Pocisk przeleciał za daleko!")
                print("Spróbuj ponownie!")
                print("-" * 30)
        except ValueError:
            print("Błąd: Podaj poprawną liczbę!")
        except KeyboardInterrupt:
            break

