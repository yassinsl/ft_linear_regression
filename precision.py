import csv
from pydantic import BaseModel, Field, ValidationError
from typing import List
from predict import estimate_price
import json
from train import Theta, Data, parse_csv_to_pydantic
import matplotlib.pyplot as plt
import numpy as np
import math



if __name__ == "__main__":
    data: List[Data] = parse_csv_to_pydantic('data.csv')
    with open("data.json", "r") as file:
        data_json = json.load(file)

    thetas: Theta = Theta(theta0=data_json['theta0'], theta1=data_json['theta1'])
    print(f"Loaded thetas: {thetas}")
    prediction : List[float] = [estimate_price(data_point.km, thetas.theta0, thetas.theta1) for data_point in data]
    error : List[float] = [prediction[i] - data_point.price for i, data_point in enumerate(data)]
    # error = real - predicted
    formatted_list = [f"{num:.2f}" for num in error]
    print(f"Errors: {formatted_list}")
    # calculate average error = sum of errors / number of data points
    average_error = sum(error) / len(error)
    print(f"Average error: {average_error:.2f}")
    # calculate root mean square error
    root_square = math.sqrt(sum(e**2 for e in error) / len(error)) 
    print(f"Root mean square error: {root_square:.2f}")
    # calcu R-squared = 1 - (sum of squared errors / sum of squared total)
    total_sum_of_squares = math.fabs(1 - (sum(e**2 for e in error) / sum((data_point.price - average_error)**2 for data_point in data)))
    print(f"R-squared: {total_sum_of_squares:.2f}")
