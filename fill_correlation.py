import pandas as pd
import numpy as np
import datetime


# 所有数据存放的在一个文件夹
file_road = "./data/round2/"
# 基金数目
fund_number = 723
output_length = int(fund_number * (fund_number-1) / 2)

data = pd.read_csv(open(file_road + "feature.csv", encoding="utf-8"), low_memory=False)
column_name = data.columns

# 填充数据
for row_index in range(539, 539+60):
    begin = datetime.datetime.now()
    all_predict = []
    index = 0
    for i in range(fund_number-1):
        for j in range(i+1, fund_number):
            index += 1
            feature_name = [column_name[output_length+i], column_name[output_length+j]]
            feature = data.iloc[row_index:][feature_name]
            # feature = data.loc[row_index:, feature_name]
            correlation = np.corrcoef(feature.T)[0, 1]
            all_predict.append(correlation)
    temp = np.array(data.iloc[row_index-1, :output_length])
    all_predict = np.array(all_predict)
    all_predict = (all_predict + temp)/2
    data.iloc[row_index, 0:output_length] = all_predict
    end = datetime.datetime.now()
    print("%d run time : %s" % (row_index, end - begin))

file = open(file_road + "feature2.csv", "w", encoding="utf-8")
data.to_csv(file, index=False)
file.close()
