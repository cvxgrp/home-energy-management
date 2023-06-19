import numpy as np
import pandas as pd
import cvxpy as cp
import itertools

def peak_power_cost(z):
    if 0 <= z <= 2:
        return 83
    elif 2 < z <= 5:
        return 147
    elif 5 < z <= 10:
        return 252
    elif 10 < z <= 15:
        return 371
    else:
        return 490
    
def get_s(z_values):
    n_months = len(z_values)
    s = np.zeros((n_months, 5))
    
    for n in range(n_months):
        z = z_values[n]
        if 0 <= z <= 2:
            s[n, :] = np.array([1, 0, 0, 0, 0])
        elif 2 < z <= 5:
            s[n, :] = np.array([0, 1, 0, 0, 0])
        elif 5 < z <= 10:
            s[n, :] = np.array([0, 0, 1, 0, 0])
        elif 10 < z <= 15:
            s[n, :] = np.array([0, 0, 0, 1, 0])
        else:
            s[n, :] = np.array([0, 0, 0, 0, 1])
    return s

def get_tier(z_values):
    n_months = len(z_values)
    
    for n in range(n_months):
        z = z_values[n]
        if 0 <= z <= 2:
            tier = 0
        elif 2 < z <= 5:
            tier = 1
        elif 5 < z <= 10:
            tier = 2
        elif 10 < z <= 15:
            tier = 3
        else:
            tier = 4
    return tier


def compute_costs(tou_prices, spot_prices, power, datetime_index, N=3, round=False):
    unique_months = pd.unique(datetime_index.month)
    monthly_tou_costs = []
    monthly_spot_costs = []
    monthly_peak_costs = []
    
    for month in unique_months:
        month_mask = (datetime_index.month == month)
        
        # Compute TOU cost
        month_tou_cost = np.sum(tou_prices[month_mask] * power[month_mask])
        monthly_tou_costs.append(month_tou_cost)
        
        # Compute spot cost
        month_spot_cost = np.sum(spot_prices[month_mask] * power[month_mask])
        monthly_spot_costs.append(month_spot_cost)
        
        # Compute peak cost
        month_length = sum(month_mask)
        daily_peak_powers = [power[month_mask][i:i+24].max() for i in range(0, month_length, 24)]
        N_largest_daily_powers = sorted(daily_peak_powers, reverse=True)[:N]
        z = sum(N_largest_daily_powers) / N
        if round:
            z = np.round(z)
        month_peak_cost = peak_power_cost(z)
        monthly_peak_costs.append(month_peak_cost)
    
    # Total costs
    tou_cost = sum(monthly_tou_costs)
    spot_cost = sum(monthly_spot_costs)
    peak_cost = sum(monthly_peak_costs)
        
    return monthly_tou_costs, monthly_spot_costs, monthly_peak_costs, tou_cost, spot_cost, peak_cost


def print_cost_summary(tou_cost, spot_cost, peak_cost):
    total_cost = tou_cost + spot_cost + peak_cost
    print(f"Total cost: {total_cost:,.2f} NOK")
    print(f"\tEnergy cost related to time-of-use prices: {tou_cost:,.2f} NOK ({100 * tou_cost / total_cost:.2f}% of total cost)")
    print(f"\tEnergy cost related to day-ahead spot prices: {spot_cost:,.2f} NOK ({100 * spot_cost / total_cost:.2f}% of total cost)")
    print(f"\tPeak power cost: {peak_cost:,.2f} NOK ({100 * peak_cost / total_cost:.2f}% of total cost)\n")
    

def get_z_values(power, datetime_index, N=3):
    unique_months = pd.unique(datetime_index.month)
    z_values = []
    
    for month in unique_months:
        month_mask = (datetime_index.month == month)
        unique_days = pd.unique(datetime_index[month_mask].date)

        daily_peak_powers = []
        for day in unique_days:
            day_mask = (datetime_index.date == day)
            daily_peak_power = power[month_mask & day_mask].max()
            daily_peak_powers.append(daily_peak_power)

        N_largest_daily_powers = sorted(daily_peak_powers, reverse=True)[:N]
        z = sum(N_largest_daily_powers) / min(N, len(daily_peak_powers))
        z_values.append(z)
        
    return z_values


def predict_load(load, baseline, AR_params, t, M, L, load_min, load_max):
    past = load[t-M:t]
    past_baseline = baseline[t-M:t]
    fut_baseline = baseline[t:t+L]
    pred = np.dot(list(reversed(past - past_baseline)), AR_params)
    
    pred = pd.Series(pred, index=fut_baseline.index)
    pred += fut_baseline
    pred = np.clip(pred, load_min, load_max)
    return pred


def make_load_forecast(load_data, load_baseline, AR_params, sim_start_time, t, H, M, L, load_min, load_max):
    curr_load = np.array([load_data[sim_start_time + t]])
    baseline_AR_forecast = predict_load(load_data, load_baseline, AR_params, sim_start_time+t+1, M, L, load_min, load_max)[:H-1]
    if H > L:
        baseline_forecast = load_baseline[sim_start_time+t+1+L:sim_start_time+t+H]
    else:
        baseline_forecast = np.array([])

    load_forecast = np.concatenate((curr_load, baseline_AR_forecast, baseline_forecast))
    return load_forecast


def make_spot_price_forecast(spot_price_data, datetime_index, sim_start_time, t, H):
    current_hour = datetime_index[t].hour

    # Determine hours with known prices
    hours_with_known_prices = min(24 - current_hour if current_hour < 13 else (24 - current_hour) + 24, H)

    # Known prices
    known_prices = spot_price_data[sim_start_time + t: sim_start_time + t + hours_with_known_prices].values

    # If the horizon extends beyond known prices, repeat the minimum known price
    if hours_with_known_prices < H:
        repeated_min_price = np.repeat(np.mean(known_prices), H - hours_with_known_prices)
        known_prices = np.concatenate((known_prices, repeated_min_price))

    return known_prices


def compute_z(p, datetime_index, p_prev=[], datetime_index_prev=None, N=3):
    # Extend p and datetime_index with previous values in the current month
    if p_prev and datetime_index_prev is not None:
        datetime_index = datetime_index_prev.append(datetime_index)
        p = cp.hstack([p_prev, p])

    z_values = []

    for month in pd.unique(datetime_index.month):
        daily_max_powers = [cp.max(p[(datetime_index.date == day) & (datetime_index.month == month)]) for day in pd.unique(datetime_index[datetime_index.month == month].date)]
        num_days = len(pd.unique(datetime_index[datetime_index.month == month].date))
        z = cp.sum_largest(cp.hstack(daily_max_powers), min(N, num_days)) / min(N, num_days)
        z_values.append(z)
        
    return z_values


def optimize(load, tou_prices, spot_prices, T, datetime_index, Q=40, C=20, D=20, q_init=20, q_final=20, p_prev=[], datetime_index_prev=None, N=3, s=None, verbose=False, solver=None):
    # Define constants
    P = 20
    eta_s, eta_c, eta_d = 0.99998, 0.95, 0.95
    tier_costs = np.array([83, 147, 252, 371, 490])
    tier_thresholds = np.array([2, 5, 10, 15, 20])
    
    # Define variables
    p, c, d, q = cp.Variable(T), cp.Variable(T), cp.Variable(T), cp.Variable(T+1)

    # Define constraints
    cons = [0 <= p, p <= P, load + c == p + d,
            q[1:] == eta_s * q[:-1] + eta_c * c - (1/eta_d) * d,
            q[0] == q_init, q[-1] == q_final,
            0 <= q, q <= Q, 0 <= c, c <= C, 0 <= d, d <= D]
    
    # Define energy charges
    energy_cost = cp.sum(cp.multiply(tou_prices + spot_prices, p))

    # Define constraints and cost function related to peak power charges
    z_values = compute_z(p, datetime_index, p_prev, datetime_index_prev, N)
    peak_power_cost = 0
    for i, z in enumerate(z_values):
        if s:
            peak_power_cost += cp.matmul(tier_costs, s[i])
            cons += [z <= cp.matmul(tier_thresholds, s[i])]
        else:
            _s = cp.Variable(len(tier_costs), boolean=True)
            peak_power_cost += cp.matmul(tier_costs, _s)
            cons += [z <= cp.matmul(tier_thresholds, _s), cp.sum(_s) == 1]

    # Define total cost
    cost = energy_cost + peak_power_cost
    
    # Define problem 
    problem = cp.Problem(cp.Minimize(cost), cons)
    
    # Solve
    problem.solve(verbose=verbose, solver=solver)
    
    return p.value, q.value, c.value, d.value, cost.value, problem.status


def global_search(load, tou_prices, spot_prices, T, datetime_index, Q=40, C=20, D=20, q_init=20, q_final=20, p_prev=[], datetime_index_prev=None, N=1):
    # Filter combinations in first repetition
    filter_combinations = False
    if p_prev:
        z_curr =  get_z_values(np.array(p_prev), datetime_index_prev, N=3)
        tier_curr = get_tier(z_curr)
        filter_combinations = True
    
    # Get the combinations
    tiers = [0, 1, 2, 3, 4]
    num_months = len(pd.unique(datetime_index.month))
    combinations = np.eye(len(tiers))

    all_combinations = list(itertools.product(tiers, repeat=num_months))

    # If in the first repetition, filter the first element of the combinations
    if filter_combinations:
        all_combinations = [combination for combination in all_combinations if combination[0] >= tier_curr]

    # Convert the tiers in the combinations to their corresponding one-hot encoded numpy arrays
    one_hot_combinations = []
    for combination in all_combinations:
        one_hot_combination = tuple(combinations[tier] for tier in combination)
        one_hot_combinations.append(one_hot_combination)

    # Initialize a variable to store the best result
    best_result = None
    best_cost = np.inf  # Start with infinity so any cost will be lower

    # Solve LP for each tier combination
    for combination in one_hot_combinations:
        p, q, c, d, cost, status = optimize(load=load, tou_prices=tou_prices, spot_prices=spot_prices, T=T, datetime_index=datetime_index, q_init=q_init, p_prev=p_prev, datetime_index_prev=datetime_index_prev, N=N, s=combination)
        
        # If the problem was solved successfully and the cost is lower than the best cost found so far, update the best result
        if status == 'optimal' and cost < best_cost:
            best_result = (p, q, c, d, cost, status)
            best_cost = cost

    # After the loop, best_result contains the variables and cost for the problem that achieved the lowest cost
    p_best, q_best, c_best, d_best, cost_best, status_best = best_result
    return p_best, q_best, c_best, d_best, cost_best, status_best

