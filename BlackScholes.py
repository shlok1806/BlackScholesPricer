from numpy import exp, sqrt, log
from scipy.stats import norm

class BlackScholesPricer :
    def __init__ (self, 
        time_to_maturity : float,
        strike_price : float,
        curr_price : float ,
        volatility : float,
        risk_free_rate : float,):
        
        self.time_to_maturity = time_to_maturity
        self.strike_price  = strike_price
        self.curr_price = curr_price
        self.volatility = volatility
        self.risk_free_rate = risk_free_rate

    def run(self,) :
        time_to_maturity = self.time_to_maturity
        strike_price = self.strike_price
        curr_price = self.curr_price
        volatility = self.volatility
        risk_free_rate = self.risk_free_rate

        d1 = (log(curr_price/strike_price) + (risk_free_rate + (volatility**2/2) )* time_to_maturity) / (volatility * sqrt(time_to_maturity))
        d2 = d1 - (volatility * sqrt(time_to_maturity))

        call_option = (curr_price * norm.cdf(d1)) - (strike_price * exp(-risk_free_rate * time_to_maturity) * norm.cdf(d2))
        put_option = (strike_price * exp(-risk_free_rate*time_to_maturity)) - curr_price + call_option

        self.call_option = call_option
        self.put_option = put_option

        self.call_delta = norm.cdf(d1)
        self.put_delta = 1 - norm.cdf(d1)
        self.call_gamma = norm.pdf(1) / (strike_price * volatility * sqrt(time_to_maturity))
        self.put_gamma = self.call_gamma
        


if __name__ == "__main__" :
    time_to_matutiry = 2
    strike_price = 90
    curr_price = 100
    volatility = 0.2
    risk_free_rate = 0.05
    
    BSP = BlackScholesPricer(time_to_maturity=time_to_matutiry,strike_price=strike_price,curr_price=curr_price,volatility=volatility,risk_free_rate=risk_free_rate)
    BSP.run()
    
