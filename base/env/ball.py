# coding=utf-8

import pandas as pd
import numpy as np
import math

from base.model.document import ColorBall
from sklearn.preprocessing import StandardScaler

class DoubleColorBall(object):

    def __init__(self, start_date='2003-01-01', end_date='2020-12-31', **options):

        self.dates = []
        self.t_dates = []
        self.e_dates = []

        # Initialize scaled  data x, y.
        self.data_x = None
        self.data_y = None

        # Initialize scaled seq data x, y.
        self.seq_data_x = None
        self.seq_data_y = None

        # Initialize flag date.
        self.next_date = None
        self.iter_dates = None
        self.current_date = None

        # Initialize parameters.
        self._init_options(**options)

        # Initialize stock data.
        self._init_data(start_date, end_date)

    def _init_options(self, **options):
        try:
            # 日志输出
            self.logger = options['logger']
        except KeyError:
            self.logger = None

        try:
            # 使用序列化数据
            self.use_sequence = options['use_sequence']
        except KeyError:
            self.use_sequence = False

        try:
            # 序列的长度
            self.seq_length = options['seq_length']
        except KeyError:
            self.seq_length = 10
        finally:
            self.seq_length = self.seq_length if self.seq_length > 1 else 2

        try:
            self.training_data_ratio = options['training_data_ratio']
        except KeyError:
            self.training_data_ratio = 0.7

        try:
            self.scaler = options['scaler']
        except KeyError:
            # 作用：去均值和方差归一化。且是针对每一个特征维度来做的，而不是针对样本。
            # 使得经过处理的数据符合标准正态分布，即均值为0，标准差为1
            self.scaler = StandardScaler()

        self.doc_class = ColorBall

    def _init_data(self, start_date, end_date):
        self._init_data_frames(start_date, end_date)
        self._init_env_data()
        self._init_data_indices()

    def _init_data_frames(self, start_date, end_date):
        columns = ['red1', 'red2', 'red3', 'red4', 'red5', 'red6', 'blue']

        ball_docs = self.doc_class.get_ball_data(start_date, end_date)
        ball_dicts = [ball.to_dic() for ball in ball_docs]

        self.dates = [ball[1] for ball in ball_dicts]
        balls = [ball[2:] for ball in ball_dicts]

        self.scaler.fit(balls)
        # 数据预处理：归一化
        balls_scaled = self.scaler.transform(balls)

        self.origin_frame = pd.DataFrame(data=balls, index=self.dates, columns=columns)
        self.scaled_frame = pd.DataFrame(data=balls_scaled, index=self.dates, columns=columns)

    def _init_env_data(self):
        if not self.use_sequence:
            # 顺序数据
            # 每次计算取一天的数据
            self._init_series_data()
        else:
            # 序列数据：序列长度为 seq_length
            # 每次计算取一个序列的数据
            self._init_sequence_data()

    def _init_series_data(self):
        self.data_count = len(self.dates[: -1])
        self.bound_index = int(self.data_count * self.training_data_ratio)

        scaled_data_x, scaled_data_y = [], []

        for index, _ in enumerate(self.dates[: -1]):
            x = self.scaled_frame.iloc[index]
            # 对应的后一天的数据序列
            y = self.origin_frame.iloc[index + 1]
            # 明确行：一行 N 列
            x = np.array(x).reshape((1, -1))
            y = np.array(y)
            # Append x, y
            scaled_data_x.append(x)
            scaled_data_y.append(y)

        # Convert list to array.
        self.data_x = np.array(scaled_data_x)
        self.data_y = np.array(scaled_data_y)

    def _init_sequence_data(self):
        self.data_count = len(self.dates[: -1 - self.seq_length])
        self.bound_index = int(self.data_count * self.training_data_ratio)

        scaled_seqs_x, scaled_seqs_y = [], []

        for date_index, _ in enumerate(self.dates[: -1]):
            # Continue until valid date index.
            if date_index < self.seq_length:
                continue

            scaled_frame = self.scaled_frame
            # 获取当前日期的前 seq_length 的数据
            balls_x = scaled_frame.iloc[date_index - self.seq_length: date_index]
            data_x = np.array(balls_x)
            # Get y, y is not at date index, but plus 1. (Training Set)
            balls_y = self.origin_frame.iloc[date_index + 1]
            data_y = np.array(balls_y)

            scaled_seqs_x.append(data_x)
            scaled_seqs_y.append(data_y)

        # Convert seq from list to array.
        self.seq_data_x = np.array(scaled_seqs_x)
        self.seq_data_y = np.array(scaled_seqs_y)

    def _init_data_indices(self):
        # Calculate indices range.
        self.data_indices = np.arange(0, self.data_count)
        self.t_data_indices = self.data_indices[:self.bound_index]
        self.e_data_indices = self.data_indices[self.bound_index:]

    def get_batch_data(self, batch_size=32):
        batch_indices = np.random.choice(self.t_data_indices, batch_size)
        if not self.use_sequence:
            batch_x = self.data_x[batch_indices]
            batch_y = self.data_y[batch_indices]
        else:
            batch_x = self.seq_data_x[batch_indices]
            batch_y = self.seq_data_y[batch_indices]
        return batch_x, batch_y

    def get_test_data(self):
        if not self.use_sequence:
            test_x = self.data_x[self.e_data_indices]
            test_y = self.data_y[self.e_data_indices]
        else:
            test_x = self.seq_data_x[self.e_data_indices]
            test_y = self.seq_data_y[self.e_data_indices]
        return test_x, test_y

    def get_origin_frame(self):
        return self.origin_frame

    def get_scaled_frame(self):
        return self.scaled_frame

    @property
    def count(self):
        return len(self.dates)

    @property
    def data_dim(self):
        data_dim = self.scaled_frame.shape[1]
        return data_dim
