# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd


# 所有数据存放的在一个文件夹
file_road = "./data/round2/"
if __name__ == "__main__":
    fund_return1 = pd.read_csv(open(file_road + "train_fund_return.csv", encoding="utf-8"), index_col=0)
    fund_return1.select_dtypes(float)
    fund_return2 = pd.read_csv(open(file_road + "test_fund_return.csv", encoding="utf-8"), index_col=0)
    fund_return = pd.concat([fund_return1, fund_return2], axis=1)
    fund_return = fund_return.transpose()

    fund_benchmark1 = pd.read_csv(open(file_road + "train_fund_benchmark_return.csv", encoding="utf-8"), index_col=0)
    fund_benchmark2 = pd.read_csv(open(file_road + "test_fund_benchmark_return.csv", encoding="utf-8"), index_col=0)
    fund_benchmark = pd.concat([fund_benchmark1, fund_benchmark2], axis=1)
    fund_benchmark = fund_benchmark.transpose()

    index_return1 = pd.read_csv(open(file_road + "train_index_return.csv", encoding="GB2312"), index_col=0)
    index_return2 = pd.read_csv(open(file_road + "test_index_return.csv", encoding="GB2312"), index_col=0)
    index_return = pd.concat([index_return1, index_return2], axis=1)
    index_return = index_return.transpose()

    correlation1 = pd.read_csv(open(file_road + "train_correlation.csv", encoding="utf-8"), index_col=0)
    correlation2 = pd.read_csv(open(file_road + "test_correlation.csv", encoding="utf-8"), index_col=0)
    correlation = pd.concat([correlation1, correlation2], axis=1)
    correlation = correlation.transpose()

    length = len(fund_return)
    correlation = correlation.reset_index()

    fund_return = fund_return.reset_index()
    correlation.drop(labels=["index"], axis=1, inplace=True)
    correlation = pd.concat([correlation, fund_return.iloc[:length]], axis=1)

    fund_benchmark = fund_benchmark.reset_index()
    fund_benchmark.drop(labels=["index"], axis=1, inplace=True)
    correlation = pd.concat([correlation, fund_benchmark.iloc[:length]], axis=1)

    index_return = index_return.reset_index()
    index_return.drop(labels=["index"], axis=1, inplace=True)
    correlation = pd.concat([correlation, index_return.iloc[:length]], axis=1)

    correlation.loc[:, "index"] = pd.to_datetime(correlation["index"])
    correlation.loc[:, "weekday"] = correlation["index"].apply(lambda x: x.weekday())
    correlation.loc[:, "date_diff"] = correlation['index'].diff(-1).apply(lambda x: abs(x.days))
    correlation.iloc[-1, -1] = 3

    correlation.loc[:, "weekday"] = (correlation["weekday"]-np.min(correlation["weekday"]))/(np.max(correlation["weekday"]) - np.min(correlation["weekday"]))
    correlation.loc[:, "date_diff"] = (correlation["date_diff"]-np.min(correlation["date_diff"]))/(np.max(correlation["date_diff"]) - np.min(correlation["date_diff"]))
    correlation.drop(labels=["index"], axis=1, inplace=True)
    file = open(file_road + "feature.csv", "w", encoding="utf-8")
    correlation.to_csv(file, index=False)
    file.close()
