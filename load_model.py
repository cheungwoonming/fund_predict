# -*- coding: UTF-8 -*-
import csv
import datetime
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
import tool


# 所有数据存放的在一个文件夹
file_road = "./data/round2/"
# GPU 配置
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 1.0
set_session(tf.Session(config=config))

# 残差单元个数
res_number = 3
# 使用多少天的数据当做一条记录
day_len = 1

# 读取数据集
data = pd.read_csv(open(file_road + "feature2.csv", "r", encoding="utf-8"), low_memory=False)
data = np.array(data).astype(float)

print("load data succeed")

# 基金数目
fund_length = 723
# 输出目标维度
target_length = int(fund_length * (fund_length-1) / 2)

predict_feature = data[-day_len:, target_length:]
predict_feature, predict_target = tool.construct_input_data(predict_feature, fund_len=fund_length, day_len=day_len)

# 读取模型，做线上预测，需要修改模型名称
model = load_model("./CheckPoint/best_model.hdf5", custom_objects={"my_loss2": tool.my_loss2})
print("load model succeed")

test_predict = model.predict(predict_feature)
reader = csv.reader(open(file_road + "submit_example_2.csv", "r", encoding="utf-8"))
file = open("./result/best_result.csv", "w", encoding="utf-8", newline="")
writer = csv.writer(file)
for index, row in enumerate(reader):
    if index != 0:
        row[1] = str(round(test_predict[0][index - 1], 4))
    writer.writerow(row)
file.close()
