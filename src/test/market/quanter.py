QUANTER_STATE_BUY = 0
QUANTER_STATE_QUIT = 1
QUANTER_STATE_SELL = 2


class Quanter:
    __initial_money = 100000
    __money = __initial_money
    __future = 0
    __ops = [('init', __initial_money)]
    __gains = []
    __days = []

    def assets(self, price):
        return self.__money + price * self.__future

    def buy(self, date, price):
        temp_future = self.__money // price
        self.__future += temp_future
        self.__money -= price * temp_future
        print("buy on {}, price = {}".format(date, price))

    def sell(self, date, price):
        temp_future = self.__money // price
        self.__future -= temp_future
        self.__money += temp_future * price
        print("sell on {}, price = {}".format(date, price))

    def quit(self, date, price):
        self.__money += self.__future * price
        self.__future = 0
        self.__ops[len(self.__ops):] = [[date, self.assets(price)]]

        curr_op = self.__ops[-1]
        last_op = self.__ops[-2]
        print("quit on {}, assets = {}, gain = {}".format(
            date, self.assets(price), ((curr_op[1] - last_op[1]) / last_op[1])
        ))

    @property
    def state(self):
        if self.__future == 0:
            return QUANTER_STATE_QUIT
        elif self.__future > 0:
            return QUANTER_STATE_BUY
        elif self.__future < 0:
            return QUANTER_STATE_SELL

    def finish(self, order_id, start_date, end_date, days, close):
        gain = (self.assets(close) - self.__initial_money) / self.__initial_money
        self.__days.append(days)
        self.__gains.append(gain)
        print("{}, {} - {}, days = {}, assets = {}, gain = {}".format(order_id, start_date, end_date, days, self.assets(close), gain))
        print("---------------------------------------------------------------------------------------------------")
        self.__money = self.__initial_money
        self.__future = 0
        self.__ops = [('init', self.__initial_money)]

    def finish_all(self):
        gain = 1
        for i in self.__gains:
            gain *= (1 + i)
        days_sum = sum(self.__days)
        print("{} days, gain {}".format(days_sum, gain))
