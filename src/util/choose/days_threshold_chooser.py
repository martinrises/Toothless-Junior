from src.data.getter.getter import DataGetter
import numpy as np
import random
import src.nn.config as config
import csv


def compute_change_rate(days, threshold):

    config.DAYS = days
    config.THRESHOLD = threshold

    getter = DataGetter()
    records = getter.data
    labels = [np.argmax(i.label) for i in records]

    # get random 1000 samples
    index = random.randint(0, len(labels) - 1000)
    data = labels[index: index + 1000]

    change_times = 0
    up_times = 0
    down_times = 0
    shake_times = 0

    last_label = data[0]
    for i in data:
        if last_label != i:
            change_times += 1
        if i == 0:
            up_times += 1
        elif i == 1:
            shake_times += 1
        else:
            down_times += 1
        last_label = i
    min_times = min((up_times, down_times, shake_times))
    rate = change_times * sum((up_times, down_times, shake_times)) / (min_times if min_times != 0 else 0.0000001)
    print("day = {}, threshold = {}, rate = {}".format(days, threshold, rate))
    return rate, change_times, up_times, down_times, shake_times


def save_to_csv(rates):
    all_result = rates
    with open(config.ROOT_PATH + "/data/choose/future/by/pick_day_threshold.csv", "r") as result_file:
        record_in_csv = list(csv.reader(result_file))
        if len(record_in_csv) != 0:
            for item in record_in_csv:
                item[2] = float(item[2])
            all_result.extend(record_in_csv)

    all_result = sorted(all_result, key=lambda item: item[2])  # sort

    # write into the file
    with open(config.ROOT_PATH + "/data/choose/future/by/pick_day_threshold.csv", "w", newline='') as result_file:
        csv.writer(result_file).writerows(all_result)


def get_change_rate():

    thresholds = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                  0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1)
    rates = []
    for day in range(3, 50):
        for threshold in thresholds:
            item = (day, threshold)
            rate = compute_change_rate(day, threshold)
            item += rate
            rates.append(item)

        if (day + 1) % 1 == 0:
            save_to_csv(rates)
            rates = []


get_change_rate()

