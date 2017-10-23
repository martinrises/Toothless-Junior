import pandas as pd
from src.data.entity.OriginRecord import OriginRecord
import src.nn.config as config
import os


def read_csv(path=config.ROOT_PATH + '/data/daily_price.csv'):
    """
    read origin data
    :param path:
    :return:
    """

    f = open(path)
    df = pd.read_csv(f)
    data = df.iloc[:, :].values
    f.close()
    return data


def get_origin_records(path=config.ROOT_PATH + '/data/daily_price.csv'):
    f = open(path)
    df = pd.read_csv(f)
    length = df.shape[0]
    records = []
    for i in range(length):
        records.append(OriginRecord(df.iloc[i, 0],
                                    df.open[i],
                                    df.close[i],
                                    df.high[i],
                                    df.low[i],
                                    df.volume[i],
                                    df.total_turnover[i]))
    f.close()
    return records


def get_future_records(path=config.ROOT_PATH + "/data/future/by/"):
    file_names = os.listdir(path)
    records = {}
    for f_name in file_names:
        records[f_name] = get_origin_records(path + f_name)
    return records
