import pandas as pd
import csv
import numpy as np


# 融合多个卷积网络的结果
def merge_conv_result():
    result1 = pd.read_csv(open("./result/result1.csv", "r", encoding="utf-8"))
    result2 = pd.read_csv(open("./result/result2.csv", "r", encoding="utf-8"))
    result3 = pd.read_csv(open("./result/result3.csv", "r", encoding="utf-8"))
    result4 = pd.read_csv(open("./result/result0.csv", "r", encoding="utf-8"))

    result1["value"] = (np.array(result1["value"]) + np.array(result2["value"]) + np.array(result3["value"]) + np.array(result4["value"]))/4
    result1.to_csv("./result/merge_conv_result.csv", index=False, float_format="%.5f")


# 融合卷积网络的结果和最后一天的结果，比例为7:3
def merge_last_result():
    reader = csv.reader(open("./result/rnn_corr_v1.csv", "r", encoding="utf-8"))
    df = pd.read_csv(open("./data/round2/test_correlation.csv", "r", encoding="utf-8"))

    with open("./result/merge_result_19.csv", "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        last = np.array(df.iloc[:, -5:]).astype(float)
        for index, row in enumerate(reader):
            if index == 0:
                writer.writerow(row)
            else:
                temp = np.mean(last[index-1], axis=0)*0.3 + float(row[1])*0.7
                writer.writerow([row[0], str(round(temp, 4))])


# 最终融合两人最好的结果
def merge_best_result():
    result1 = pd.read_csv("./result/best_result_1.csv")
    result2 = pd.read_csv("./result/sub_1.csv")
    result2["value"] = np.array(result1.iloc[:, 1])*0.3 + np.array(result2.iloc[:, 1])*0.7
    result2.to_csv("./result/last_merge_best_result_37.csv", index=False, float_format="%.5f")


if __name__ == "__main__":
    # merge_conv_result()
    merge_last_result()

    # merge_best_result()
