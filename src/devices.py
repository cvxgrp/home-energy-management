import cvxpy as cp
import numpy as np

from .network import Device, Terminal
 
class Grid(Device):
    def __init__(self, price_buy, price_peak_demand, prev_peak_demand, month_slices, price_sell=0):
        super().__init__([Terminal()])
        self.price_buy = price_buy
        self.price_peak_demand = price_peak_demand
        self.prev_peak_demand = prev_peak_demand
        self.month_slices = month_slices
        self.price_sell = price_sell

    @property
    def cost(self):
        p = self.terminals[0].power_var
        cost_energy_usage = cp.sum(cp.maximum(cp.multiply(-self.price_buy, p), cp.multiply(-self.price_sell, p)))
        prev_peak_demands = [self.prev_peak_demand] + [0] * (len(self.month_slices) - 1)
        cost_peak_demand = cp.sum([self.price_peak_demand * cp.pos(cp.max(-p[month]) - prev_peak_demand) for month, prev_peak_demand in zip(self.month_slices, prev_peak_demands)])
        cost = cost_energy_usage + cost_peak_demand
        return cost

class Load(Device):
    def __init__(self, power):
        super().__init__([Terminal()])
        self.power = power

    @property
    def constraints(self):
        constraints = [self.terminals[0].power_var == self.power]
        return constraints
 
class Storage(Device):
    def __init__(self, discharge_max, charge_max, energy_init, energy_final, energy_max, len_interval=1.0):
        super().__init__([Terminal()])
        self.discharge_max = discharge_max
        self.charge_max = charge_max
        self.energy_init = energy_init
        self.energy_max = energy_max
        self.energy_final = energy_final
        self.len_interval = len_interval
        self.energy = None

    @property
    def constraints(self):
        p = self.terminals[0].power_var
        if self.energy is None:
            self.energy = cp.Variable(self.terminals[0].power_var.shape)
        
        constraints = [self.energy[0] - self.energy_init - p[0] * self.len_interval == 0,
                       p >= -self.discharge_max,
                       p <= self.charge_max,
                       self.energy <= self.energy_max,
                       self.energy >= 0,
                       self.energy[-1] >= self.energy_final]
        
        if p.shape[0] > 1:
            constraints += [cp.diff(self.energy) == p[1:] * self.len_interval]

        return constraints
