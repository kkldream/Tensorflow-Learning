import random
import pandas as pd
import header

def generate(num):
    tag_arr = []
    day_arr = []
    num_arr = [0] * 12
    while True:
        rand = random.randint(1, 365)
        rand_timeStr = header.dayToTimeStr(rand)
        rand_tag = header.timeStrToConstellation(rand_timeStr)
        # rand_tag_name = tag_indexs[rand_tag]
        # print(rand, rand_timeStr, rand_tag, rand_tag_name)
        if num_arr[rand_tag] < num:
            tag_arr.append(rand_tag)
            day_arr.append(rand)
            num_arr[rand_tag] += 1
        if sum(num_arr) >= num * 12:
            break
    grades = {
        "tag": tag_arr,
        "day": day_arr
    }
    df = pd.DataFrame(grades)
    return df

if __name__ == "__main__":
    data = generate(100)
    data.to_csv('data2.csv')
