import csv
from pydantic import BaseModel, Field, ValidationError
from typing import List
from predict import estimate_price
import json

class Theta(BaseModel):
    theta0: float = Field(alias="theta0")
    theta1: float = Field(alias="theta1")
    def __str__(self):
        return f"Theta(theta0={self.theta0}, theta1={self.theta1})"

class Data(BaseModel):
    km: int = Field(alias="km")
    price: int

    def __str__(self):
        return f"Data(km={self.km}, price={self.price})"

def parse_csv_to_pydantic(FilePath: str)-> List[Data]:
    data_list = []
    try:
        with open(FilePath, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None:
                print(f"Error: The CSV file '{FilePath}' is empty or has no header.")
                exit(1)
            for row in reader:
                try:
                    data = Data.model_validate(row)
                    if data is not None:
                        data_list.append(data)
                except ValidationError as e:
                    print(f"Error validating row: {e}")
                    exit(1)
    except FileNotFoundError:
        print(f"Error: File '{FilePath}' not found.")
        exit(1)
    except Exception as e:
        print(f"Error: An error occurred while reading '{FilePath}': {e}")
        exit(1)
    return data_list

def main():
    data : List[Data] = parse_csv_to_pydantic('data.csv')
    thetas : Theta = Theta(theta0=0.0, theta1=0.0)
    len_data = 0
    for d in data:
        if d.km < 0 or d.price < 0:
            print("Error: Invalid data found in 'data.csv'.")
            exit(1)
        len_data = len_data + 1
    if len_data == 0:
        print("Error: No valid data found in 'data.csv'.")
        exit(1)
    max_km = max(d.km for d in data)
    min_km = min(d.km for d in data)
    normalized_data = [(d.km - min_km) / (max_km - min_km) for d in data]
    learining_rate = 0.5
    error = [0] * len_data

    for _ in range(1000):
        #calculate error for each data point "error positive go down, error negative go up"
        error = [
            estimate_price(normalized_data[i], thetas.theta0, thetas.theta1)
            - data[i].price
            for i in range(len_data)
        ]

        tmp_theta0 = (
            learining_rate * sum(error[i] for i in range(len_data)) / len_data
        )
        tmp_theta1 = (
            learining_rate
            * sum(error[i] * normalized_data[i] for i in range(len_data))
            / len_data
        )

        thetas.theta0 = thetas.theta0 - tmp_theta0
        thetas.theta1 = thetas.theta1 - tmp_theta1
    #θ1 = θ1 / (max − min)
    thetas.theta1 = thetas.theta1 / (max_km - min_km) 
    #θ0 = θ0 − θ1 * min / (max − min) 
    thetas.theta0 = thetas.theta0 - (thetas.theta1 * min_km)
    with open("data.json", "w", encoding="utf-8") as file:
        json.dump({
            "theta0": thetas.theta0,
            "theta1": thetas.theta1
        }, file)
    print(f"Training completed. Final parameters: {thetas}")


if __name__ == "__main__":
    main()