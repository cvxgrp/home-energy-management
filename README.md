# Home Energy Management with Dynamic Tariffs and Tiered Peak Power Charges

This repository accompanies our [paper](https://web.stanford.edu/~boyd/papers/hem.html) and provides code and data to replicate all results.

## Abstract
We consider a simple home energy system consisting of a (net) load, an energy storage device, and a grid connection. We focus on minimizing the cost for grid power that includes a time-varying usage price and a tiered peak power charge that depends on the average of the largest N daily powers over a month. When the loads and prices are known, the optimal operation of the storage device can be found by solving a mixed-integer linear program (MILP). This prescient charging policy is not implementable in practice, but it does give a bound on the best performance possible. We propose a simple model predictive control (MPC) method that relies on simple forecasts of future prices and loads. The MPC problem is also an MILP, but it can be solved directly as a linear program (LP) using simple enumeration of the tiers for the current and next months, and so is fast and reliable. Numerical experiments on real data from a home in Trondheim, Norway, show that the MPC policy achieves a cost that is only 1.7% higher than the prescient performance bound.

## Citing
If you want to reference our paper in your research, please consider citing us by using the following BibTeX:
```
@misc{perezpineiro2023home,
      title={Home Energy Management with Dynamic Tariffs and Tiered Peak Power Charges}, 
      author={David Pérez-Piñeiro and Sigurd Skogestad and Stephen Boyd},
      year={2023},
      eprint={2307.07580},
      archivePrefix={arXiv},
      primaryClass={math.OC}
}
```
