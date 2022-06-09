import numpy as np
import pandas as pd

def get_std_arr(std, num):
    arr = np.random.normal(std[0], std[1], num)
    arr = np.clip(arr, 0, 100)
    return arr

data_num = 500
labels = ["literature", "science"]
col_name = ["group", "chinese", "english", "math", "nature", "society", "physics"]
liberal_std = [(80, 10), (65, 10), (30, 10), (40, 10), (70, 15), (50, 10)]
science_std = [(40, 10), (60, 15), (80, 10), (80, 15), (40, 20), (80, 10)]

data = {}
for i, col in enumerate(col_name):
    if i == 0:
        data[col] = [labels[0]] * (data_num // 2) + [labels[1]] * (data_num // 2)
    else:
        concatenate = np.concatenate([
            get_std_arr(liberal_std[i - 1], data_num // 2),
            get_std_arr(science_std[i - 1], data_num // 2)
        ])
        data[col_name[i]] = concatenate

dataset = pd.DataFrame(data)

filename = f"dataset.csv"
dataset.to_csv(filename)
print(f"Saved {filename}")