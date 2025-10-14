import math
import datetime

def calculate_biorhythm(day, cycle_length):
    """Oblicza wartość biorytmu dla danego dnia i długości cyklu"""
    return math.sin(2 * math.pi * day / cycle_length)

def get_day_of_life(birth_date):
    """Oblicza ile dni minęło od urodzenia do dzisiaj"""
    today = datetime.date.today()
    days_of_life = (today - birth_date).days
    return days_of_life

def get_tomorrow_trend(day, cycle_length):
    """Sprawdza trend na następny dzień (True jeśli będzie lepiej)"""
    today_value = calculate_biorhythm(day, cycle_length)
    tomorrow_value = calculate_biorhythm(day + 1, cycle_length)
    return tomorrow_value > today_value

def main():
    print("=== KALKULATOR BIORYTMÓW ===")
    print("Oblicz swoje biorytmy i dowiedz się, jak się dziś czujesz!")
    name = input("Podaj swoje imię: ")
    print("Podaj swoją datę urodzenia:")
    year = int(input("Rok (np. 1990): "))
    month = int(input("Miesiąc (1-12): "))
    day = int(input("Dzień: "))
    birth_date = datetime.date(year, month, day)
    days_of_life = get_day_of_life(birth_date)
    print(f"Witaj {name}!")
    print(f"Dzisiaj jest twój {days_of_life}. dzień życia!")
    physical_score = calculate_biorhythm(days_of_life, 23)
    emotional_score = calculate_biorhythm(days_of_life, 28)
    intellectual_score = calculate_biorhythm(days_of_life, 33)
    print("=== TWOJE BIORYTMY NA DZISIAJ ===")
    print(f"Fizyczny: {physical_score:.3f}")
    print(f"Emocjonalny: {emotional_score:.3f}")
    print(f"Intelektualny: {intellectual_score:.3f}")
    print("=== ANALIZA ===")
    if physical_score > 0.5:
        print("🎉 Gratulacje! Twój fizyczny biorytm jest wysoki - masz dziś dużo energii!")
    elif physical_score < -0.5:
        print("💪 Twój fizyczny biorytm jest niski - możesz czuć się zmęczony.")
        if get_tomorrow_trend(days_of_life, 23):
            print("   Nie martw się. Jutro będzie lepiej pod względem fizycznym!")
    if emotional_score > 0.5:
        print("😊 Gratulacje! Twój emocjonalny biorytm jest wysoki - świetny nastrój!")
    elif emotional_score < -0.5:
        print("😔 Twój emocjonalny biorytm jest niski - możesz czuć się przygnębiony.")
        if get_tomorrow_trend(days_of_life, 28):
            print("   Nie martw się. Jutro będzie lepiej pod względem emocjonalnym!")
    if intellectual_score > 0.5:
        print("🧠 Gratulacje! Twój intelektualny biorytm jest wysoki - twój umysł jest ostry!")
    elif intellectual_score < -0.5:
        print("🤔 Twój intelektualny biorytm jest niski - możesz mieć problemy z koncentracją.")
        if get_tomorrow_trend(days_of_life, 33):
            print("   Nie martw się. Jutro będzie lepiej pod względem intelektualnym!")
    print(f"Dziękujemy za skorzystanie z kalkulatora biorytmów, {name}!")
    print("Pamiętaj, że biorytmy to tylko teoria - najważniejsze jest to, jak Ty się czujesz!")

if __name__ == "__main__":
    main()
