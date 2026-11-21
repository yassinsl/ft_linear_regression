import csv
from pydantic import BaseModel, Field, ValidationError
from typing import List
from predict import estimate_price
import json
from train import Theta, Data, parse_csv_to_pydantic
import matplotlib.pyplot as plt
import numpy as np



if __name__ == "__main__":
    data: List[Data] = parse_csv_to_pydantic('data.csv')
    with open("data.json", "r") as file:
        data_json = json.load(file)

    thetas: Theta = Theta(theta0=data_json['theta0'], theta1=data_json['theta1'])
    print(f"Loaded thetas: {thetas}")
    x = np.array([d.km for d in data])
    x_points = np.linspace(min(x), max(x), 100)
    y_points = [estimate_price(x, thetas.theta0, thetas.theta1) for x in x_points]

    plt.plot(x_points, y_points, color='blue', label='Regression Line')
    for d in data:
        plt.scatter(d.km, d.price, color='red', label='Data Points' if d == data[0] else "")
    plt.xlabel('Mileage (km)')
    plt.ylabel('Price')
    plt.title('Linear Regression: Mileage vs Price')
    plt.legend()
    plt.grid()
    plt.show()

