import pandas as pd
import numpy as np
import csv
import datetime


def get_score(y_true, y_pred):
    mae = np.mean(np.abs(y_pred-y_true), axis=-1)
    tmape = np.mean(np.abs((y_pred-y_true)/(1.5-y_true)))
    return np.mean((2/(2+mae+tmape))**2)


data = pd.read_csv(open("./data/feature.csv", encoding="utf-8"))
column_name = data.columns

index = 0
all_target = []
all_predict = []
row_index = 538
for i in range(199):
    for j in range(i+1, 200):
        target = data.iloc[row_index, index]
        index += 1
        feature_name = [column_name[19900+i], column_name[19900+j]]
        # feature_name = [column_name[19900+200+i], column_name[19900+200+j]]
        feature = data.loc[row_index:row_index+61, feature_name]
        # feature = np.array(feature)
        # feature = np.concatenate((feature, feature[-20:]))
        correlation = np.corrcoef(feature.T)[0, 1]
        all_target.append(target)
        all_predict.append(correlation)
print("score: ", get_score(np.array(all_target), np.array(all_predict)))

