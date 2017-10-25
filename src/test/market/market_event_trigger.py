from src.test.market.quanter import Quanter
import src.test.market.quanter as quanter
import numpy as np
import src.nn.config as config

MARKET_UP = 0
MARKET_SHAKE = 1
MARKET_DOWN = 2

DAYS_THRESHOLD = 3

class MarketEventTrigger:
    __last_market_state = 1
    __quanter = Quanter()

    def trigger_market_event(self, order_id,  predictions, datas):
        for i in range(len(datas) - 1):
            prediction = predictions[i]
            data = datas[i]
            next_day_data = datas[i+1]
            state = np.argmax(prediction)
            self.on_market_event(state, np.argmax(data.label), data.date, data.close, next_day_data.open, i)

            if self.__last_market_state != state:
                self.on_market_state_change(self.__last_market_state, state, data.date, next_day_data.open, i + 1)

            self.__last_market_state = state
        self.finish(order_id, datas[0].date, datas[-1].date, len(datas), datas[-1].close)

    def on_market_event(self, state, target_state, date, close, next_day_open, index):
        if not config.FIX_MISTAKE:
            return
        if self.__quanter.state == quanter.QUANTER_STATE_BUY and state == MARKET_UP:
            if index - self.__quanter.last_op_index >= DAYS_THRESHOLD and self.__quanter.last_op_price <= close:  # check whether if bought more than 3 days and price is lower than itself 3 days ago
                print("buy wrong")
                self.__quanter.quit(date, next_day_open, index)
        elif self.__quanter.state == quanter.QUANTER_STATE_SELL and state == MARKET_DOWN:
            if index - self.__quanter.last_op_index >= DAYS_THRESHOLD and self.__quanter.last_op_price >= close:  # check whether if sold more than 3 days and price is higher than itself 3 days ago
                print("sell wrong")
                self.__quanter.quit(date, next_day_open, index)

    def reset_state(self, date, price, index):
        if self.__quanter.state != quanter.QUANTER_STATE_QUIT:
            self.__quanter.quit(date, price, index)

    def on_market_state_change(self, old_state, state, date, next_day_open, next_day_index):
        if state == MARKET_UP:
            self.reset_state(date, next_day_open, next_day_index)
            self.__quanter.buy(date, next_day_open, next_day_index)
        elif state == MARKET_DOWN:
            self.reset_state(date, next_day_open, next_day_index)
            self.__quanter.sell(date, next_day_open, next_day_index)
        else:
            self.__quanter.quit(date, next_day_open, next_day_index)

    def finish(self, order_id, start_date, end_date, days, close):
        self.__quanter.finish(order_id, start_date, end_date, days, close)

    def trigger_finish_all(self):
        self.__quanter.finish_all()
