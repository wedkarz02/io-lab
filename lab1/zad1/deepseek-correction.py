import math
import sys
import datetime
from typing import Optional, Callable, List, Tuple

def calculate_days_since_birth(year: int, month: int, day: int) -> int:
    """Calculate the number of days since the given birth date."""
    today = datetime.date.today()
    birth_date = datetime.date(year, month, day)
    if birth_date > today:
        raise ValueError("Birth date cannot be in the future")
    return (today - birth_date).days

def default_wave(t: int, factor: int) -> float:
    """Calculate the default wave value using sine function."""
    return math.sin((2 * math.pi * t) / factor)

def physical_wave(t: int) -> float:
    """Calculate the physical biorhythm wave (23-day cycle)."""
    return default_wave(t, 23)

def emotional_wave(t: int) -> float:
    """Calculate the emotional biorhythm wave (28-day cycle)."""
    return default_wave(t, 28)

def intellectual_wave(t: int) -> float:
    """Calculate the intellectual biorhythm wave (33-day cycle)."""
    return default_wave(t, 33)

def get_mood_status(score: float) -> Optional[bool]:
    """Determine mood status based on biorhythm score."""
    if score >= 0.5:
        return True
    elif score <= -0.5:
        return False
    return None

def get_mood_description(is_happy: Optional[bool], day_type: str = "today") -> str:
    """Generate a descriptive message for the mood status."""
    mood_map = {
        True: "excellent",
        False: "challenging",
        None: "neutral"
    }
    return f"{day_type}: {mood_map[is_happy]}"

def format_function_name(func_name: str) -> str:
    """Convert function name to human-readable format."""
    return func_name.replace("_", " ").title()

def get_user_birthdate() -> Tuple[int, int, int]:
    """Safely get and validate user birthdate input."""
    print("Please enter your birthdate:")
    try:
        year = int(input("Year (YYYY): "))
        month = int(input("Month (1-12): "))
        day = int(input("Day (1-31): "))
        datetime.date(year, month, day)
        return year, month, day
    except (ValueError, TypeError) as e:
        print(f"Invalid date: {e}")
        sys.exit(1)

def analyze_biorhythm(wave_function: Callable[[int], float], 
                     days: int, 
                     name: str) -> dict:
    """Analyze biorhythm for a given wave function."""
    func_name = wave_function.__name__
    display_name = format_function_name(func_name)
    score_today = wave_function(days)
    score_tomorrow = wave_function(days + 1)
    mood_today = get_mood_status(score_today)
    mood_tomorrow = get_mood_status(score_tomorrow)
    return {
        'name': display_name,
        'score_today': score_today,
        'score_tomorrow': score_tomorrow,
        'mood_today': mood_today,
        'mood_tomorrow': mood_tomorrow
    }

def display_results(name: str, analyses: List[dict]):
    """Display the biorhythm analysis results in a formatted way."""
    print(f"\n{'='*50}")
    print(f"BIORHYTHM ANALYSIS FOR: {name.upper()}")
    print(f"Date: {datetime.date.today().strftime('%Y-%m-%d')}")
    print(f"{'='*50}")
    for analysis in analyses:
        print(f"\n{analysis['name']} Biorhythm:")
        print(f"  Score: {analysis['score_today']:.3f}")
        print(f"  Status: {get_mood_description(analysis['mood_today'])}")
        if analysis['mood_today'] is False:
            print(f"  Tomorrow's outlook: {get_mood_description(analysis['mood_tomorrow'], 'tomorrow')}")

def main():
    """Main function to run the biorhythm analysis."""
    try:
        name = input("Enter your name: ").strip()
        if not name:
            print("Name cannot be empty.")
            return
        year, month, day = get_user_birthdate()
        days_since_birth = calculate_days_since_birth(year, month, day)
        print(f"\nHello {name}! You are {days_since_birth} days young!")
        wave_functions = [physical_wave, emotional_wave, intellectual_wave]
        analyses = []
        for wave_func in wave_functions:
            analysis = analyze_biorhythm(wave_func, days_since_birth, name)
            analyses.append(analysis)
        display_results(name, analyses)
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

