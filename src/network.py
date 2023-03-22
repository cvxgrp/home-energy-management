import cvxpy as cp

class Terminal:
    @property
    def power_var(self):
        return self._power

    @property
    def power(self):
        return self._power.value

    def init_problem(self, time_horizon):
        self._power = cp.Variable(time_horizon)

class Net:
    def __init__(self, terminals):
        self.terminals = terminals
        self.constraints = []

    def init_problem(self, time_horizon):
        for terminal in self.terminals:
            terminal.init_problem(time_horizon)

        self.constraints = [cp.sum([t.power_var for t in self.terminals]) == 0]

class Device:
    def __init__(self, terminals):
        self.terminals = terminals

    @property
    def cost(self):
        return 0

    @property
    def constraints(self):
        return []

    def init_problem(self, time_horizon):
        for terminal in self.terminals:
            terminal.init_problem(time_horizon)

class Group(Device):
    def __init__(self, devices, nets, terminals=[]):
        super().__init__(terminals)
        self.devices = devices
        self.nets = nets
        self._constraints = []  # change this line

    def init_problem(self, time_horizon):
        for device in self.devices:
            device.init_problem(time_horizon)

        for net in self.nets:
            net.init_problem(time_horizon)

        self._constraints = (  # change this line
            [c for device in self.devices for c in device.constraints] +
            [c for net in self.nets for c in net.constraints]
        )

    @property
    def constraints(self):  # add this property method
        return self._constraints

    def solve(self, **kwargs):
        problem = cp.Problem(cp.Minimize(cp.sum([device.cost for device in self.devices])), self.constraints)
        problem.solve(**kwargs)
        return problem.value
    
    
    
# class Group(Device):
#     def __init__(self, devices, nets, terminals=[]):
#         super().__init__(terminals)
#         self.devices = devices
#         self.nets = nets
#         self._constraints = []
#         self.status = None  # Add this line

#     def init_problem(self, time_horizon):
#         for device in self.devices:
#             device.init_problem(time_horizon)

#         for net in self.nets:
#             net.init_problem(time_horizon)

#         self._constraints = (
#             [c for device in self.devices for c in device.constraints] +
#             [c for net in self.nets for c in net.constraints]
#         )

#     @property
#     def constraints(self):
#         return self._constraints

#     def solve(self, **kwargs):
#         problem = cp.Problem(cp.Minimize(cp.sum([device.cost for device in self.devices])), self.constraints)
#         problem.solve(**kwargs)
#         self.status = problem.status  # Update the status attribute after solving
#         return problem.value
