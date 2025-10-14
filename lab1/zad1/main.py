import math
import sys
import datetime

def calculate_days_since_birdth(year, month, day):
    now = datetime.date.today()
    days_since_birdth = (now - datetime.date(year, month, day)).days
    return days_since_birdth

def default_wave(t, factor):
    return math.sin((2 * math.pi * t) / factor)

def physical_wave(t):
    return default_wave(t, 23)

def emotional_wave(t):
    return default_wave(t, 28)

def intelectual_wave(t):
    return default_wave(t, 33)

def rank_score(score):
    if score >= 0.5: return True
    elif score <= -0.5: return False
    return None

if __name__ == "__main__":
    name = input("name: ")
    try:
        year = int(input("year: "))
        month = int(input("month: "))
        day = int(input("day: "))
    except:
        print("invalid data")
        sys.exit()

    days_since_birdth = calculate_days_since_birdth(year, month, day)

    for fn in [physical_wave, emotional_wave, intelectual_wave]:
        score = fn(days_since_birdth)
        print(f"result of {fn.__name__.replace("_", " ")}: {score}")

        is_happy = rank_score(score)
        if is_happy:
            print(f"{fn.__name__} result: you're happy")
        elif is_happy is not None:
            print(f"{fn.__name__} result: you're not happy")
            score_next_day = fn(days_since_birdth + 1)
            is_happy_next_day = rank_score(score_next_day)
            if is_happy_next_day:
                print(f"{fn.__name__} result for next day: you're happy")
            elif is_happy_next_day is not None:
                print(f"{fn.__name__} result for next day: you're not happy")
            else:
                print(f"{fn.__name__} result for next day: neutral")
        else:
            print(f"{fn.__name__} result: neutral")

# c) Trudno porównać czas mojego pisania:
# - niespecjalnie zwracałem uwagi na czas,
# - chwilę zajęło setupowanie środowiska, instalowanie LSP itd.
# - typowe siedzenie i myślenie "a jakby to zrobić lepiej" zamiast napisać to co jest proste i działa

