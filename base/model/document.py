# coding=utf-8

from mongoengine import Document
from mongoengine.fields import StringField, FloatField, DateTimeField, IntField


class Stock(Document):
    # 股票代码
    code = StringField(required=True)
    # 交易日
    date = DateTimeField(required=True)
    # 开盘价
    open = FloatField()
    # 最高价
    high = FloatField()
    # 最低价
    low = FloatField()
    # 收盘价
    close = FloatField()
    # 成交量
    volume = FloatField()
    # 成交金额
    amount = FloatField()
    # 涨跌幅
    p_change = FloatField()
    # 价格变动
    price_change = FloatField()
    # 5日均价
    ma5 = FloatField()
    # 10日均价
    ma10 = FloatField()
    # 20日均价
    ma20 = FloatField()
    # 5日均量
    v_ma5 = FloatField()
    # 10日均量
    v_ma10 = FloatField()
    # 20日均量
    v_ma20 = FloatField()
    # 换手率
    turnover = FloatField()

    meta = {
        'indexes': [
            'code',
            'date',
            ('code', 'date')
        ]
    }

    def save_if_need(self):
        return self.save() if len(self.__class__.objects(code=self.code, date=self.date)) < 1 else None

    def to_state(self):
        stock_dic = self.to_mongo()
        stock_dic.pop('_id')
        stock_dic.pop('code')
        stock_dic.pop('date')
        return stock_dic.values()

    def to_dic(self):
        stock_dic = self.to_mongo()
        stock_dic.pop('_id')
        return stock_dic.values()

    @classmethod
    def get_k_data(cls, code, start, end):
        return cls.objects(code=code, date__gte=start, date__lte=end).order_by('date')

    @classmethod
    def exist_in_db(cls, code):
        return True if cls.objects(code=code)[:1].count() else False


class Future(Document):
    # 合约代码
    code = StringField(required=True)
    # 交易日
    date = DateTimeField(required=True)
    # 开盘价
    open = FloatField()
    # 最高价
    high = FloatField()
    # 最低价
    low = FloatField()
    # 收盘价
    close = FloatField()
    # 成交量
    volume = FloatField()

    meta = {
        'indexes': [
            'code',
            'date',
            ('code', 'date')
        ]
    }

    def save_if_need(self):
        return self.save() if len(self.__class__.objects(code=self.code, date=self.date)) < 1 else None

    def to_state(self):
        stock_dic = self.to_mongo()
        stock_dic.pop('_id')
        stock_dic.pop('code')
        stock_dic.pop('date')
        return stock_dic.values()

    def to_dic(self):
        stock_dic = self.to_mongo()
        stock_dic.pop('_id')
        return stock_dic.values()

    @classmethod
    def get_k_data(cls, code, start, end):
        return cls.objects(code=code, date__gte=start, date__lte=end).order_by('date')

    @classmethod
    def exist_in_db(cls, code):
        return True if cls.objects(code=code)[:1].count() else False


class ColorBall(Document):
    # 期数
    phase = StringField(required=True)
    # 开奖日期
    date = DateTimeField(required=True)
    # 红1
    red1 = IntField()
    # 红2
    red2 = IntField()
    # 红3
    red3 = IntField()
    # 红4
    red4 = IntField()
    # 红5
    red5 = IntField()
    # 红6
    red6 = IntField()
    # 蓝
    blue = IntField()

    meta = {
        'indexes': [
            'phase',
            'date',
            ('phase', 'date')
        ]
    }

    def save_if_need(self):
        return self.save() if len(self.__class__.objects(phase=self.phase, date=self.date)) < 1 else None

    def to_state(self):
        ball_dic = self.to_mongo()
        ball_dic.pop('_id')
        ball_dic.pop('phase')
        ball_dic.pop('date')
        return ball_dic.values()

    def to_dic(self):
        ball_dic = self.to_mongo()
        ball_dic.pop('_id')
        return ball_dic.values()

    @classmethod
    def get_ball_data(cls, start, end):
        return cls.objects(date__gte=start, date__lte=end).order_by('phase')

    @classmethod
    def exist_in_db(cls, phase):
        return True if cls.objects(phase=phase)[:1].count() else False
