import csv
from pydantic import BaseModel, Field, ValidationError
from typing import List

class Data(BaseModel):
    km: int = Field(alias="km")
    price: int
     
    def __str__(self):
        return f"Data(km={self.km}, price={self.price})"

def parse_csv_to_pydantic(FilePath: str)-> List[Data]:
    data_list = []
    with open(FilePath, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                data = Data.model_validate(row)
                data_list.append(data)
            except ValidationError as e:
                print(f"Error validating row: {e}")
                exit(1)
    return data_list

def main():
    data : List[Data] = parse_csv_to_pydantic('data.csv')
    for d in data:
        print(d)

if __name__ == "__main__":
    main()