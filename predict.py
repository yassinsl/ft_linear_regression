import json
import math
import sys

THETAS_FILE = "Thetas_case/thetas.json"


def is_valid_number(value) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)

def load_model_parameters(file_path: str) -> tuple[float, float]:
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"'{file_path}' not found, model not trained yet (theta0 = 0, theta1 = 0).")
        return 0.0, 0.0
    except json.JSONDecodeError:
        sys.exit(f"Error: '{file_path}' is not valid JSON.")
    except OSError as e:
        sys.exit(f"Error: cannot read '{file_path}': {e}")
    if not isinstance(data, dict):
        sys.exit(f"Error: '{file_path}' must contain a JSON object.")
    for key in ("theta0", "theta1"):
        if key not in data:
            sys.exit(f"Error: '{key}' is missing in '{file_path}'.")
        if not is_valid_number(data[key]):
            sys.exit(f"Error: '{key}' must be a finite number.")
    return float(data["theta0"]), float(data["theta1"])

def ask_mileage() -> int:
    while True:
        try:
            text = input("Enter mileage (km): ")
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit("Bye!")
        try:
            mileage = int(text)
        except ValueError:
            print("Please enter a valid integer for mileage.")
            continue
        if mileage < 0:
            print("Mileage cannot be negative.")
            continue
        return mileage

def estimate_price(mileage: float, theta0: float, theta1: float) -> float:
    return theta0 + (theta1 * mileage)

def main():
    theta0, theta1 = load_model_parameters(THETAS_FILE)
    mileage = ask_mileage()
    price = estimate_price(mileage, theta0, theta1)
    print(f"Estimated price for {mileage} km: {price:.2f}")
    if price < 0:
        print("Warning: negative price, this mileage is outside the range of the data.")

if __name__ == "__main__":
    main()