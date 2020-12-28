# coding=utf-8

import pandas as pd
import numpy as np
import math

from base.env.trader import Trader
from base.model.document import Stock, Future
from sklearn.preprocessing import StandardScaler


class Market(object):

    Running = 0
    Done = -1

    def __init__(self, codes, start_date="2008-01-01", end_date="2018-01-01", **options):

        # 初始化股票代码
        self.codes = codes
        self.index_codes = []
        self.state_codes = []

        # 初始化股票日期
        self.dates = []
        self.t_dates = []
        self.e_dates = []

        # 初始化数据
        self.origin_frames = dict()
        self.scaled_frames = dict()

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
            # future 期货
            # stock  股票
            self.m_type = options['market']
        except KeyError:
            # 股票
            self.m_type = 'stock'

        try:
            # 初始现金
            self.init_cash = options['cash']
        except KeyError:
            self.init_cash = 100000

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
            # 使用归一化数据
            self.use_normalized = options['use_normalized']
        except KeyError:
            self.use_normalized = True

        try:
            self.mix_trader_state = options['mix_trader_state']
        except KeyError:
            self.mix_trader_state = True

        try:
            self.mix_index_state = options['mix_index_state']
        except KeyError:
            self.mix_index_state = False
        finally:
            if self.mix_index_state:
                self.index_codes.append('sh')

        try:
            # 序列的长度
            self.seq_length = options['seq_length']
        except KeyError:
            self.seq_length = 5
        finally:
            self.seq_length = self.seq_length if self.seq_length > 1 else 2

        try:
            self.training_data_ratio = options['training_data_ratio']
        except KeyError:
            self.training_data_ratio = 0.7

        try:
            scaler = options['scaler']
        except KeyError:
            # 作用：去均值和方差归一化。且是针对每一个特征维度来做的，而不是针对样本。
            # 使得经过处理的数据符合标准正态分布，即均值为0，标准差为1
            scaler = StandardScaler

        # 状态码为：股票码 和 指数码
        self.state_codes = self.codes + self.index_codes
        # 为每个状态码分配一个 scaler，预处理归一化
        self.scaler = [scaler() for _ in self.state_codes]
        # 初始化 trader
        self.trader = Trader(self, cash=self.init_cash)
        # 区分股票或期货
        self.doc_class = Stock if self.m_type == 'stock' else Future

    def _init_data(self, start_date, end_date):
        self._init_data_frames(start_date, end_date)
        self._init_env_data()
        self._init_data_indices()

    def _validate_codes(self):
        # 用来训练的数据不为0
        if not self.state_code_count:
            raise ValueError("Codes cannot be empty.")
        # 用来训练的数据必须存在
        for code in self.state_codes:
            if not self.doc_class.exist_in_db(code):
                raise ValueError("Code: {} not exists in database.".format(code))

    def _init_data_frames(self, start_date, end_date):
        # Remove invalid codes first.
        self._validate_codes()
        # Init columns and data set.
        # 使用的数据为：开盘价、最高价、最低价、收盘价、成交量
        columns, dates_set = ['open', 'high', 'low', 'close', 'volume'], set()
        # Load data.
        for index, code in enumerate(self.state_codes):
            # Load instrument docs by code.
            # 获取日线数据
            instrument_docs = self.doc_class.get_k_data(code, start_date, end_date)
            # Init instrument dicts.
            # 每日数据转换成字典
            instrument_dicts = [instrument.to_dic() for instrument in instrument_docs]
            # Split dates.
            # 拿到日期列表
            dates = [instrument[1] for instrument in instrument_dicts]
            # Split instruments.
            # 拿到交易数据：open/high/low/close/volume/...
            instruments = [instrument[2:] for instrument in instrument_dicts]
            # Update dates set.
            # 所有股票日期集合，去除重复日期
            dates_set = dates_set.union(dates)
            # Build origin and scaled frames.
            scaler = self.scaler[index]
            scaler.fit(instruments)
            # 数据预处理：归一化
            instruments_scaled = scaler.transform(instruments)
            # 原始数据，只要：开盘价、最高价、最低价、收盘价、成交量
            origin_frame = pd.DataFrame(data=instruments, index=dates, columns=columns)
            # 归一化后的数据，只要：开盘价、最高价、最低价、收盘价、成交量
            scaled_frame = pd.DataFrame(data=instruments_scaled, index=dates, columns=columns)
            # Build code - frame map.
            # 保存数据
            self.origin_frames[code] = origin_frame
            self.scaled_frames[code] = scaled_frame
        
        # Init date iter.
        # 日期排序
        self.dates = sorted(list(dates_set))
        # Rebuild index.
        for code in self.state_codes:
            origin_frame = self.origin_frames[code]
            scaled_frame = self.scaled_frames[code]
            # 重建数据，将空数据的日期按 method 填充。bfill: 使用后一个数据填充，ffill：使用前一个数据填充
            self.origin_frames[code] = origin_frame.reindex(self.dates, method='bfill')
            self.scaled_frames[code] = scaled_frame.reindex(self.dates, method='bfill')

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
        # Calculate data count.
        self.data_count = len(self.dates[: -1])
        # Calculate bound index.
        # 计算学习使用的数据边界 0 - bound_index 用来训练，bound_index - end 用来测试
        self.bound_index = int(self.data_count * self.training_data_ratio)
        # Init scaled_x, scaled_y.
        scaled_data_x, scaled_data_y = [], []
        for index, _ in enumerate(self.dates[: -1]):
            # Get current x, y.
            # 某一天所有股票的数据序列
            x = [self.scaled_frames[code].iloc[index] for code in self.state_codes]
            # 对应的后一天的数据序列
            y = [self.scaled_frames[code].iloc[index + 1] for code in self.state_codes]
            # Convert x, y to array.
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
        # Calculate data count.
        self.data_count = len(self.dates[: -1 - self.seq_length])
        # Calculate bound index.
        self.bound_index = int(self.data_count * self.training_data_ratio)
        # Init seqs_x, seqs_y.
        scaled_seqs_x, scaled_seqs_y = [], []
        # Scale to valid dates.
        for date_index, _ in enumerate(self.dates[: -1]):
            # Continue until valid date index.
            if date_index < self.seq_length:
                continue
            data_x, data_y = [], []
            for index, code in enumerate(self.state_codes):
                # Get scaled frame by code.
                scaled_frame = self.scaled_frames[code]
                # Get instrument data x.
                # 获取当前日期的前 seq_length 的数据
                instruments_x = scaled_frame.iloc[date_index - self.seq_length: date_index]
                data_x.append(np.array(instruments_x))
                # Get instrument data y.
                if index < date_index - 1:
                    if date_index < self.bound_index:
                        # Get y, y is not at date index, but plus 1. (Training Set)
                        instruments_y = scaled_frame.iloc[date_index + 1]['close']
                    else:
                        # Get y, y is at date index. (Test Set)
                        instruments_y = scaled_frame.iloc[date_index + 1]['close']
                    data_y.append(np.array(instruments_y))
            # Convert list to array.
            data_x = np.array(data_x)
            data_y = np.array(data_y)
            seq_x = []
            seq_y = data_y
            # Build seq x, y.
            for seq_index in range(self.seq_length):
                seq_x.append(data_x[:, seq_index, :].reshape((-1)))
            # Convert list to array.
            seq_x = np.array(seq_x)
            scaled_seqs_x.append(seq_x)
            scaled_seqs_y.append(seq_y)
        # Convert seq from list to array.
        self.seq_data_x = np.array(scaled_seqs_x)
        self.seq_data_y = np.array(scaled_seqs_y)

    def _init_data_indices(self):
        # Calculate indices range.
        self.data_indices = np.arange(0, self.data_count)
        # Calculate train and eval indices.
        self.t_data_indices = self.data_indices[:self.bound_index]
        self.e_data_indices = self.data_indices[self.bound_index:]
        # Generate train and eval dates.
        self.t_dates = self.dates[:self.bound_index]
        self.e_dates = self.dates[self.bound_index:]

    def _origin_data(self, code, date):
        date_index = self.dates.index(date)
        return self.origin_frames[code].iloc[date_index]

    def _scaled_data_as_state(self, date):
        if not self.use_sequence:
            data = self.data_x[self.dates.index(date)]
            if self.mix_trader_state:
                trader_state = self.trader.scaled_data_as_state()
                data = np.insert(data, -1, trader_state).reshape((1, -1))
            return data
        else:
            return self.seq_data_x[self.dates.index(date)]

    def reset(self, mode='train'):
        # Reset trader.
        self.trader.reset()
        # Reset iter dates by mode.
        self.iter_dates = iter(self.t_dates) if mode == 'train' else iter(self.e_dates)
        try:
            self.current_date = next(self.iter_dates)
            self.next_date = next(self.iter_dates)
        except StopIteration:
            raise ValueError("Reset error, dates are empty.")
        # Reset baseline.
        self._reset_baseline()
        return self._scaled_data_as_state(self.current_date)

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

    def forward(self, stock_code, action_code):
        # Check Trader.
        self.trader.remove_invalid_positions()
        self.trader.reset_reward()
        # Get stock data.
        stock = self._origin_data(stock_code, self.current_date)
        stock_next = self._origin_data(stock_code, self.next_date)
        # Execute transaction.
        action = self.trader.action_by_code(action_code)
        action(stock_code, stock, 100, stock_next)
        # Init episode status.
        episode_done = self.Running
        # Add action times.
        self.trader.action_times += 1
        # Update date if need.
        if self.trader.action_times == self.code_count:
            self._update_profits_and_baseline()
            try:
                self.current_date, self.next_date = self.next_date, next(self.iter_dates)
            except StopIteration:
                episode_done = self.Done
            finally:
                self.trader.action_times = 0
        # Get next state.
        state_next = self._scaled_data_as_state(self.current_date)
        # Return s_n, r, d, info.
        return state_next, self.trader.reward, episode_done, self.trader.cur_action_status

    def _update_profits_and_baseline(self):
        prices = [self._origin_data(code, self.current_date).close for code in self.codes]
        baseline_profits = np.dot(self.stocks_holding_baseline, np.transpose(prices)) - self.trader.initial_cash
        policy_profits = self.trader.profits
        self.trader.history_baselines.append(baseline_profits)
        self.trader.history_profits.append(policy_profits)

    def _reset_baseline(self):
        # Calculate cash piece.
        cash_piece = self.init_cash / self.code_count
        # Get stocks data.
        stocks = [self._origin_data(code, self.current_date) for code in self.codes]
        # Init stocks baseline.
        self.stocks_holding_baseline = [int(math.floor(cash_piece / stock.close)) for stock in stocks]

    @property
    def code_count(self):
        return len(self.codes)

    @property
    def index_code_count(self):
        return len(self.index_codes)

    @property
    def state_code_count(self):
        return len(self.state_codes)

    @property
    def data_dim(self):
        data_dim = self.state_code_count * self.scaled_frames[self.codes[0]].shape[1]
        if not self.use_sequence:
            if self.mix_trader_state:
                data_dim += (2 + self.code_count)
        return data_dim
