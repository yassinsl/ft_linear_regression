import csv
from pydantic import BaseModel, Field, ValidationError
from typing import List
from predict import estimate_price

class Theta(BaseModel):
    theta0: float = Field(alias="theta0")
    theta1: float = Field(alias="theta1")
    def __str__(self):
        """Return a readable representation of the theta values."""
        return f"Theta(theta0={self.theta0}, theta1={self.theta1})"

class Data(BaseModel):
    km: int = Field(alias="km")
    price: int

    def __str__(self):
        """Return a readable representation of the data point."""
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
    for d in data:
        print(d)

if __name__ == "__main__":
    main()