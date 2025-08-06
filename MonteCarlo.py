import numpy as np 
import matplotlib.pyplot as plt

class MonteCarlo : 
    def __init__ (self, 
        time_to_maturity : float,
        curr_price : float ,
        volatility : float,
        risk_free_rate : float,
        num_of_sim_paths : int, 
        num_of_steps : int, 
        expected_return : float
        ):
        
        self.time_to_maturity = time_to_maturity
        self.curr_price = curr_price
        self.volatility = volatility
        self.risk_free_rate = risk_free_rate
        self.num_of_sim_paths = num_of_sim_paths
        self.num_of_steps = num_of_steps
        self.dt = time_to_maturity/num_of_steps
        self.expected_return = expected_return

    def run(self,) :
        time_to_maturity = self.time_to_maturity
        curr_price = self.curr_price
        volatility = self.volatility
        risk_free_rate = self.risk_free_rate
        num_of_sim_paths = self.num_of_sim_paths
        num_of_steps = self.num_of_steps
        dt = time_to_maturity/num_of_steps
        expected_return = self.expected_return
        
        
        price_paths = np.zeros((num_of_sim_paths, num_of_steps + 1))
        price_paths[:, 0] = curr_price
        
        random_shocks = np.random.normal(0, 1, size=(num_of_sim_paths, num_of_steps))
        
        for i in range(1, num_of_steps + 1) :
            price_paths[:, i] = price_paths[:, i - 1] * np.exp((expected_return - 0.5 * volatility**2) * dt + volatility* np.sqrt(dt) * random_shocks[:, i - 1])

        final_price = price_paths[:, -1]
        return price_paths

