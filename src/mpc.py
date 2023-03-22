import numpy as np
import pandas as pd
import cvxpy as cp
import tqdm

from .helper_functions import create_month_slices

class OptimizationError(Exception):
    """Error due to infeasibility or numerical problems during optimization."""
    pass

class MPC:
    """
    The MPC class represents a generic Model Predictive Control (MPC) implementation.
    """
    
    def __init__(self, make_network, params, load_data, price_data, baseline, sim_start_time):
        self.make_network = make_network
        self.params = params
        self.load_data = load_data
        self.price_data = price_data
        self.baseline = baseline
        self.sim_start_time = sim_start_time
        self.energy_stored = None
        self.power = None
        self.peak_demand = None
        self.cost_electricity_use = None
        self.cost_peak_demand = None

    def make_forecasts(self, t):
        pass

    def implement(self, t, MPC_network):
        self.energy_stored[t] = MPC_network.devices[1].energy.value[0]
        self.params["initial_storage"] = self.energy_stored[t]
        self.power[t] = -MPC_network.nets[0].terminals[0].power[0]
        self.peak_demand[t] = np.max([self.power[t], self.params["prev_peak_demand"]])

        if t == self.time_steps - 1 or self.load_data.index[self.sim_start_time+t].month != self.load_data.index[self.sim_start_time+t+1].month:
            self.params["prev_peak_demand"] = 0
        else:
            self.params["prev_peak_demand"] = self.peak_demand[t]

    def run(self, time_steps, **kwargs):
        self.time_steps = time_steps
        self.energy_stored = np.zeros(time_steps)
        self.power = np.zeros(time_steps)
        self.peak_demand = np.zeros(time_steps)

        for t in tqdm.trange(time_steps):
            if time_steps - t < self.params["T"]:
                self.params["T"] = time_steps - t

            self.make_forecasts(t)
            MPC_network = self.make_network(self.params)
            MPC_network.solve(**kwargs)
            if MPC_network.problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                raise OptimizationError(
                "failed at iteration %d, %s" % (t, MPC_network.problem.status)
            )
            self.implement(t, MPC_network)
            
        # Compute peak demand and cost
        p_grid = pd.Series(data=self.power, index=self.load_data[self.sim_start_time:self.sim_start_time+time_steps].index)
        monthly_peak_demands = p_grid.resample('M').max().values
        self.peak_demand = p_grid.groupby(p_grid.index.to_period('M')).transform('max').to_numpy()
        self.cost_electricity_use = np.sum(p_grid.values * self.price_data[self.sim_start_time:self.sim_start_time+time_steps])
        self.cost_peak_demand = np.sum(monthly_peak_demands * self.params["price_peak_demand"])

        return self.energy_stored, self.power, self.peak_demand, self.cost_electricity_use, self.cost_peak_demand
    
    # def run(self, time_steps, **kwargs):
    #     self.time_steps = time_steps
    #     self.energy_stored = np.zeros(time_steps)
    #     self.power = np.zeros(time_steps)
    #     self.peak_demand = np.zeros(time_steps)

    #     for t in tqdm.trange(time_steps):
    #         if time_steps - t < self.params["T"]:
    #             self.params["T"] = time_steps - t

    #         self.make_forecasts(t)
    #         MPC_network = self.make_network(self.params)
    #         MPC_network.solve(**kwargs)
    #         if MPC_network.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):  # Update this line
    #             raise OptimizationError(
    #                 "failed at iteration %d, %s" % (t, MPC_network.status)  # Update this line
    #             )

    #         self.implement(t, MPC_network)
            
    #         # Compute peak demand and cost
    #         p_grid = pd.Series(data=self.power, index=self.load_data[self.sim_start_time:self.sim_start_time+time_steps].index)
    #         monthly_peak_demands = p_grid.resample('M').max().values
    #         self.peak_demand = p_grid.groupby(p_grid.index.to_period('M')).transform('max').to_numpy()
    #         self.cost_electricity_use = np.sum(p_grid.values * self.price_data[self.sim_start_time:self.sim_start_time+time_steps])
    #         self.cost_peak_demand = np.sum(monthly_peak_demands * self.params["price_peak_demand"])

    #         return self.energy_stored, self.power, self.peak_demand, self.cost_electricity_use, self.cost_peak_demand

class BaselineMPC(MPC):
    """
    The BaselineMPC class is a subclass of the MPC class that uses a seasonal baseline forecast for the electricity demand.
    """
    
    def __init__(self, make_network, params, load_data, price_data, baseline, sim_start_time):
        super().__init__(make_network, params, load_data, price_data, baseline, sim_start_time)

    def make_forecasts(self, t):
        self.params["electricity_demand"] = self.baseline[self.sim_start_time+t:self.sim_start_time+t+self.params["T"]].values
        self.params["electricity_price"] = self.price_data[self.sim_start_time+t:self.sim_start_time+t+self.params["T"]].values
        self.params["month_slices"] = create_month_slices(self.load_data[self.sim_start_time+t:self.sim_start_time+t+self.params["T"]].index)


class AutoregressiveMPC(MPC):
    """
    The AutoregressiveMPC class is a subclass of the MPC class that uses a seasonal baseline in addition to an autoregressive residual model for predicting electricity demand.
    """

    def __init__(self, make_network, params, load_data, price_data, baseline, sim_start_time, autoreg_residual_params):
        super().__init__(make_network, params, load_data, price_data, baseline, sim_start_time)
        self.autoreg_residual_params = autoreg_residual_params

    def predict_load(self, p_fixed_load, baseline, autoreg_residual_params, t, M, L, K=1):
        past = p_fixed_load[t-M:t]
        past_baseline = baseline[t-M:t]
        fut_baseline = baseline[t:t+L]
        pred = list(reversed(past-past_baseline)) @ autoreg_residual_params
        pred = pd.Series(pred, index=fut_baseline.index)
        pred += fut_baseline
        pred = np.maximum(self.load_data.min(), pred)
        pred = np.minimum(self.load_data.max(), pred)
        return pred

    def make_forecasts(self, t):
        self.params["electricity_demand"] = np.concatenate([[self.load_data[self.sim_start_time+t]],
            self.predict_load(self.load_data, self.baseline, self.autoreg_residual_params, self.sim_start_time+t+1, self.params["T"], self.params["T"])[:-1]])
        self.params["electricity_price"] = self.price_data[self.sim_start_time+t:self.sim_start_time+t+self.params["T"]]
        self.params["month_slices"] = create_month_slices(self.load_data[self.sim_start_time+t:self.sim_start_time+t+self.params["T"]].index)