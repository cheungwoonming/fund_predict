from keras.layers import Input, Activation, BatchNormalization, Dense, Flatten, Dropout, Concatenate, PReLU, Add, \
    Conv2D, MaxPool2D
from keras import backend as K
from keras import Model
import numpy as np
from random import randint


# 两种loss 叠加：mae + tmape
def my_loss1(y_true, y_pred):
    mae = K.mean(K.abs(y_pred-y_true), axis=-1)
    tmape = K.mean(K.abs((y_pred-y_true)/(1.5-y_true)))
    return mae+tmape


# 只算一种loss：tmape
def my_loss2(y_true, y_pred):
    return K.mean(K.abs((y_pred-y_true)/(1.5-y_true)))


def monitor_score(y_true, y_pred):
    mae = K.mean(K.abs(y_pred-y_true), axis=-1)
    tmape = K.mean(K.abs((y_pred-y_true)/(1.5-y_true)))
    return K.mean((2/(2+mae+tmape))**2)


def get_score(y_true, y_pred):
    mae = np.mean(np.abs(y_pred-y_true), axis=-1)
    tmape = np.mean(np.abs((y_pred-y_true)/(1.5-y_true)))
    return np.mean((2/(2+mae+tmape))**2)


def get_score_each(y_true, y_pred):
    mae = np.mean(np.abs(y_pred-y_true), axis=-1)
    tmape = np.mean(np.abs((y_pred-y_true)/(1.5-y_true)))
    return (2/(2+mae+tmape))**2


def _bn_relu_conv(nb_filter, nb_row, nb_col, bn=True):
    def f(input):
        if bn:
            input = BatchNormalization(axis=3)(input)
        activation = Activation("relu")(input)
        return Conv2D(filters=nb_filter, kernel_size=(nb_row, nb_col), strides=(1, 1), padding="same")(activation)
    return f


# 两层卷积
def residual_units(nb_filter=16, nb_row=3, nb_col=3, repetition=3):
    def f(input):
        for i in range(repetition):
            residual = _bn_relu_conv(nb_filter, nb_row, nb_col)(input)
            residual = _bn_relu_conv(nb_filter, nb_row, nb_col)(residual)
            input = Add()([input, residual])
        return input
    return f


# 三层卷积，减少参数，第一层将nb_filter减小为一半
def residual_units2(nb_filter=16, nb_row=3, nb_col=3, repetition=3):
    def f(input):
        for i in range(repetition):
            residual = _bn_relu_conv(int(nb_filter/2), 1, 1)(input)
            residual = _bn_relu_conv(int(nb_filter/2), nb_row, nb_col)(residual)
            residual = _bn_relu_conv(nb_filter, 1, 1)(residual)
            input = Add()([input, residual])
        return input
    return f


def construct_conv_nn(feature_length, map_channels=2, nb_filter=16, output_length=19900, map_row=200, map_col=200, res_number =3):
    map_input = Input(shape=(map_row, map_col, map_channels))
    conv1 = Conv2D(filters=nb_filter, kernel_size=(3, 3), strides=(3, 3), padding="valid")(map_input)
    # 降维
#     conv1 = MaxPool2D(pool_size=(2, 2), padding="valid")(conv1)

    residual_output = residual_units(nb_filter=nb_filter, repetition=res_number)(conv1)

    activation = Activation("relu")(residual_output)
    conv2 = Conv2D(filters=map_channels, kernel_size=(3, 3), strides=(1, 1), padding="same")(activation)

    pool2d = MaxPool2D(pool_size=(2, 2), padding="valid")(conv2)
    map_output = Flatten()(pool2d)

    feature_input = Input(shape=(feature_length,))
    feature_output = Dense(500)(feature_input)
    feature_output = BatchNormalization()(feature_output)
    feature_output = PReLU()(feature_output)

    feature_output = Dense(1000)(feature_output)
    feature_output = BatchNormalization()(feature_output)
    feature_output = PReLU()(feature_output)

    feature_output = Concatenate()([feature_output, feature_input])

    feature_output = Dense(1000)(feature_output)
    feature_output = BatchNormalization()(feature_output)
    feature_output = PReLU()(feature_output)

    output = Concatenate()([map_output, feature_output])
    output = Dense(1000, activation="relu")(output)
    output = Dropout(0.3)(output)
    output = Dense(output_length)(output)

    model = Model(inputs=[map_input, feature_input], outputs=output)
    return model


# 把200个基金收益率序列变成一个200*200的图
def to_img(array):
    row_num, column_num = array.shape
    img = np.zeros((row_num, column_num, column_num))
    for row_index, row in enumerate(array):
        for i in range(column_num):
            for j in range(column_num):
                img[row_index, i, j] = row[j]-row[i]
    img = np.abs(img)
    # img = (img-np.min(img))/(np.max(img)-np.min(img))
    return img


# 构造网络的输入和输出，预测集不需要输入all_target，day_length为使用日期长度
def construct_input_data(all_feature, all_target=None, use_benchmark=True, fund_len=723, day_len=1):
    # 是否使用 benchmark_return 数据
    if use_benchmark:
        feature1 = all_feature[:, :fund_len]
        feature2 = all_feature[:, fund_len:fund_len*2]
        img1 = to_img(feature1)
        img2 = to_img(feature2)

        n_sample = len(img1)-day_len+1
        input_data1 = np.zeros((n_sample, fund_len, fund_len, 2*day_len))
        one_feature_len = all_feature.shape[1]
        input_data2 = np.zeros((n_sample, one_feature_len * day_len))
        for index in range(n_sample):
            for i in range(day_len):
                input_data1[index, :, :, i*2] = img1[index+i]
                input_data1[index, :, :, i*2+1] = img2[index+i]
                input_data2[index, i*one_feature_len:(i+1)*one_feature_len] = all_feature[index+i]
    else:
        feature1 = all_feature[:, :fund_len]
        img1 = to_img(feature1)
        n_sample = len(img1) - day_len + 1
        input_data1 = np.zeros((n_sample, fund_len, fund_len, day_len))
        one_feature_len = all_feature.shape[1]
        input_data2 = np.zeros((n_sample, one_feature_len * day_len))
        for index in range(n_sample):
            for i in range(day_len):
                input_data1[index, :, :, i] = img1[index + i]
                input_data2[index, i * one_feature_len:(i + 1) * one_feature_len] = all_feature[index + i]

    if all_target is not None:
        all_target = all_target[day_len - 1:, :]

    return [input_data1, input_data2], all_target


# 保证每个epoch都用到了所有的训练数据
def generate_batch_data(x, y, batch_size, steps_per_epoch):
    steps = -1
    while True:
        # 每个epoch之后都重新打乱顺序
        if steps == (steps_per_epoch-1):
            data_num = len(x[0])
            index = list(range(data_num))
            np.random.shuffle(index)
            x = [x[0][index], x[1][index]]
            y = y[index]
            steps = -1
        steps += 1
        yield [x[0][steps * batch_size:(steps + 1) * batch_size], x[1][steps * batch_size:(steps + 1) * batch_size]], y[steps * batch_size:(steps + 1) * batch_size]


# 每个epoch存在部分训练数据使用了多次
def generate_batch_data2(x, y, batch_size, steps_per_epoch):
    while True:
        steps = randint(0, steps_per_epoch)
        yield [x[0][steps * batch_size:(steps + 1) * batch_size], x[1][steps * batch_size:(steps + 1) * batch_size]], y[steps * batch_size:(steps + 1) * batch_size]
