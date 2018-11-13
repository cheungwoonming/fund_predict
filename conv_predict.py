# -*- coding: UTF-8 -*-
import csv
import datetime
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import optimizers
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import tool


# 所有数据存放的在一个文件夹
file_road = "./data/round2/"
# GPU 配置
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 1.0
set_session(tf.Session(config=config))
# 网络参数
nb_filter = 8
batch_size = 32
# 是否使用 benchmark_return
use_benchmark = False
# 残差单元个数
res_number = 3
# 使用多少天的数据当做一条记录
day_len = 1

# 使用最后几天的数据作为测试集
test_size = 5

begin = datetime.datetime.now()
now_date = time.strftime('%Y-%m-%d', time.localtime(time.time()))

# 读取数据集
data = pd.read_csv(open(file_road + "feature2.csv", "r", encoding="utf-8"), low_memory=False)
data = np.array(data).astype(float)

print("load succeed")
# 已知相关性的数据集长度
use_data_length = 539
# 基金数目
fund_length = 723
# 输出目标维度
target_length = int(fund_length * (fund_length-1) / 2)

train_feature = data[:use_data_length-test_size, target_length:]
train_target = data[:use_data_length-test_size, :target_length]

test_feature = data[use_data_length-test_size-day_len+1:use_data_length, target_length:]
test_target = data[use_data_length-test_size-day_len+1:use_data_length, :target_length]


# 额外增加使用多少天的皮尔逊系数
def add_data(old_feature, old_target, ext_length=20):
    ext_feature = data[use_data_length: use_data_length+ext_length, target_length:]
    ext_target = data[use_data_length: use_data_length+ext_length, :target_length]
    new_feature = np.vstack((old_feature, ext_feature))
    new_target = np.vstack((old_target, ext_target))
    return new_feature, new_target


# 增大训练集
train_feature, train_target = add_data(train_feature, train_target)

# 包含了图和一维特征
train_feature, train_target = tool.construct_input_data(train_feature, train_target, fund_len=fund_length, use_benchmark=use_benchmark, day_len=day_len)
test_feature, test_target = tool.construct_input_data(test_feature, test_target, fund_len=fund_length, use_benchmark=use_benchmark, day_len=day_len)
predict_feature = data[-day_len:, target_length:]
predict_feature, predict_target = tool.construct_input_data(predict_feature, fund_len=fund_length, use_benchmark=use_benchmark, day_len=day_len)

# 增加星期五的数据量，初赛有用，复赛效果不好
if False:
    index = np.where(train_feature[1][:, -2] == 1.0)
    x_ext0 = train_feature[0][index]
    x_ext1 = train_feature[1][index]
    y_ext = train_target[index]
    for i in range(3):
        train_feature[0] = np.concatenate((train_feature[0], x_ext0))
        train_feature[1] = np.concatenate((train_feature[1], x_ext1))
        train_target = np.concatenate((train_target, y_ext))

    # 打乱训练集的顺序
    data_num = len(train_feature[0])
    index = list(range(data_num))
    np.random.shuffle(index)

    train_feature = [train_feature[0][index], train_feature[1][index]]
    train_target = train_target[index]

print("construct feature succeed")

# 一维特征的特征数量
feature_length = train_feature[1].shape[1]
output_length = train_target.shape[1]
map_row = fund_length
map_col = fund_length
map_channels = day_len
if use_benchmark:
    map_channels = 2*day_len


model = tool.construct_conv_nn(feature_length=feature_length, map_channels=map_channels, nb_filter=nb_filter,
                               output_length=output_length, map_row=map_row, map_col=map_col, res_number=res_number)
adam = optimizers.Adam(lr=0.001)
model.compile(optimizer=adam, loss=tool.my_loss2, metrics=[tool.monitor_score])
print("compile succeed")
train_scores = []
test_scores = []

# 训练
epochs = 15
for i in range(25):
    model.fit(train_feature, train_target, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(test_feature, test_target))
    # steps_per_epoch = use_data_length//batch_size
    # model.fit_generator(tool.generate_batch_data(train_feature, train_target, batch_size, steps_per_epoch), epochs=epochs, workers=1, steps_per_epoch=steps_per_epoch, validation_data=(test_feature, test_target))

    # train_predict = model.predict_generator(train_feature)
    # score = tool.get_score(train_target, train_predict)
    print("NO.%d:" % (i + 1))
    # print("train score: ", score)
    # train_scores.append(score)

    test_predict = model.predict(test_feature)
    score = tool.get_score(test_target, test_predict)
    print("test score: ", score)
    print("test score foreach ", tool.get_score_each(test_target, test_predict))
    test_scores.append(score)


# check_point 只保存结果最好的模型
check_point = ModelCheckpoint(filepath="./CheckPoint/cp-{epoch:02d}e-train_score_{monitor_score:.4f}-val_score_{val_monitor_score:.4f}.hdf5", monitor='val_monitor_score',verbose=0, save_best_only=True, save_weights_only=False, mode='max', period=1)
# check_point = ModelCheckpoint(filepath="./CheckPoint/best_model.hdf5", monitor='val_monitor_score', verbose=0, save_best_only=True, save_weights_only=False, mode='max', period=1)
epochs = 50
model.fit(train_feature, train_target, epochs=epochs, batch_size=batch_size, verbose=1,
          callbacks=[check_point],
          validation_data=(test_feature, test_target))
end = datetime.datetime.now()
print("run time : %s" % (end-begin))

# 读取模型，做线上预测
# model = load_model("./CheckPoint/best_model.hdf5", custom_objects={"my_loss2": my_loss2})
test_predict = model.predict(predict_feature)
reader = csv.reader(open(file_road + "submit_example_2.csv", "r", encoding="utf-8"))
file = open("./result/best_result.csv", "w", encoding="utf-8", newline="")
writer = csv.writer(file)
for index, row in enumerate(reader):
    if index != 0:
        row[1] = str(round(test_predict[0][index - 1], 4))
    else:
        row = ["ID", "value"]
    writer.writerow(row)
file.close()
