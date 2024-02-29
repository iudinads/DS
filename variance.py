
from typing import List
import math
import matplotlib.pyplot as plt
import numpy as np
import statistics


def _sum_of_squares(data : List[float]) -> float: # реализация суммы квадратов
    sum = 0
    for i in range(len(data)):
        sum += data[i] ** 2
    return sum
        
def _mean(data : List[float]) -> float:
    x_bar = statistics.mean(data) # среднее значение
    return [x - x_bar for x in data]

def variance(data : List[float]) -> float: #дисперсия 
    assert len(data) >= 2
    n = len(data)
    deviations = _mean(data) #отклонение
    return _sum_of_squares(deviations) / (n - 1)

def standart_deviation(data : List[float]) -> float: # стандартное отклонение
    return math.sqrt(variance(data))

def covariance(xs : List[float], ys : List[float]) -> float: #отклонение двух переменных от своих средних
    assert len(xs) == len(ys)
    return np.dot(_mean(xs), _mean(ys)) / (len(xs) - 1)

def correlation(xs : List[float], ys : List[float]) -> float:
    stdev_x = standart_deviation(xs)
    stdev_y = standart_deviation(ys)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(xs, ys) / (stdev_x * stdev_y)
    else: 
        return 0
    
x_arr = [0, 2, 1, 6, 3]
y_arr = [100, 101, 102, 103, 104]

print(correlation(x_arr, y_arr))  #корркляция от - 1(вообще не зависимы) до +1
print(covariance([0, 2, 1, 6, 3], [100, 101, 102, 103, 104]))

