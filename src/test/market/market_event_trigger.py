from src.test.market.quanter import Quanter
import tensorflow as tf
import src.test.market.quanter as quanter

MARKET_UP = 0
MARKET_SHAKE = 1
MARKET_DOWN = 2


class MarketEventTrigger:
    __last_market_state = 1
    __quanter = Quanter()

    def trigger_market_event(self, predictions, datas):
        for i in range(len(datas) - 1):
            prediction = predictions[i]
            data = datas[i]
            next_day_data = datas[i+1]
            state = tf.argmax(prediction, 1)
            self.on_market_event(state, data.date, next_day_data.open)

            if self.__last_market_state != state:
                self.on_market_state_change(self.__last_market_state, state, data.date, next_day_data.open)

            self.__last_market_state = state
        self.finish(datas[-1].date, datas[-1].close)

    def on_market_event(self, state, date, next_day_open):
        pass

    def reset_state(self, date, price):
        if self.__quanter.state != quanter.QUANTER_STATE_QUIT:
            self.__quanter.quit(date, price)

    def on_market_state_change(self, old_state, state, date, next_day_open):
        if state == MARKET_UP:
            self.reset_state(date, next_day_open)
            self.__quanter.buy(date, next_day_open)
        elif state == MARKET_DOWN:
            self.reset_state(date, next_day_open)
            self.__quanter.sell(date, next_day_open)
        else:
            self.__quanter.quit(date, next_day_open)

    def finish(self, date, close):
        self.__quanter.finish(date, close)
