import numpy as np
import pandas as pd
import cvxpy as cp

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
    
def get_y(z_values):
    n_months = len(z_values)
    y = np.zeros((n_months, 5))
    
    for n in range(n_months):
        z = z_values[n]
        if 0 <= z <= 2:
            y[n, :] = np.array([1, 0, 0, 0, 0])
        elif 2 < z <= 5:
            y[n, :] = np.array([0, 1, 0, 0, 0])
        elif 5 < z <= 10:
            y[n, :] = np.array([0, 0, 1, 0, 0])
        elif 10 < z <= 15:
            y[n, :] = np.array([0, 0, 0, 1, 0])
        else:
            y[n, :] = np.array([0, 0, 0, 0, 1])
    return y


def compute_costs(tou_prices, spot_prices, power, datetime_index, N=3):
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
        month_peak_cost = peak_power_cost(np.round(z))
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


def get_z_hourly_values(power, datetime_index, N=3):
    unique_months = pd.unique(datetime_index.month)
    z_hourly_values = []
    
    for month in unique_months:
        month_mask = (datetime_index.month == month)
        month_length = sum(month_mask)
        daily_peak_powers = [power[month_mask][i:i+24].max() for i in range(0, month_length, 24)]
        N_largest_daily_powers = sorted(daily_peak_powers, reverse=True)[:N]
        z = sum(N_largest_daily_powers) / N
        z_hourly_values.extend([z] * month_length)
        
    return z_hourly_values

def get_z_values(power, datetime_index, N=3):
    unique_months = pd.unique(datetime_index.month)
    z_values = []
    
    for month in unique_months:
        month_mask = (datetime_index.month == month)
        month_length = sum(month_mask)
        daily_peak_powers = [power[month_mask][i:i+24].max() for i in range(0, month_length, 24)]
        N_largest_daily_powers = sorted(daily_peak_powers, reverse=True)[:N]
        z = sum(N_largest_daily_powers) / N
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


# def predict_load(load, baseline, AR_params, t, M, L):
#     past = load[t-M:t]
#     past_baseline = baseline[t-M:t]
#     fut_baseline = baseline[t:t+L]
#     pred = np.dot(list(reversed(past - past_baseline)), AR_params)
    
#     pred = pd.Series(pred, index=fut_baseline.index)
#     pred += fut_baseline
#     return pred

# def make_load_forecast(load_data, load_baseline, AR_params, sim_start_time, t, H, M, L, K=1, sigma_residual_errors=None):
#     MIN = load_data.min()
#     MAX = load_data.max()
    
#     if sigma_residual_errors is None:
#         curr_load = np.array([load_data[sim_start_time + t]])
#         baseline_AR_forecast = predict_load(load_data, load_baseline, AR_params, sim_start_time+t+1, M, L)[:-1]
#         scenarios = np.matrix(np.concatenate((curr_load, baseline_AR_forecast)))
#     else:
#         scenarios = np.hstack([np.matrix([load_data[sim_start_time+t]]*K).T, np.random.multivariate_normal(predict_load(load_data, load_baseline, AR_params, sim_start_time+t+1, M, L), sigma_residual_errors, K)[:,:-1]])
    
#     scenarios = np.clip(scenarios, MIN, MAX)
#     load_forecast = scenarios.T

#     if load_forecast.shape[0] < H:
#         extra_baseline = load_baseline[sim_start_time+t+L:sim_start_time+t+H]
#         extra_baseline_matrix = np.tile(extra_baseline, (K, 1)).T
#         load_forecast = np.concatenate((load_forecast, extra_baseline_matrix), axis=0)

#     return load_forecast



# def make_load_forecast(load_data, load_baseline, AR_params, sim_start_time, t, H, M, L, load_min, load_max):
#     curr_load = np.array([load_data[sim_start_time + t]])
#     baseline_forecast = load_baseline[sim_start_time+t+1:sim_start_time+t+H]

#     load_forecast = np.concatenate((curr_load, baseline_forecast))
#     return load_forecast







# def make_spot_price_forecast(spot_price_data, spot_price_baseline, datetime_index, sim_start_time, t, H):
#     current_hour = datetime_index[t].hour

#     # Determine hours with known prices
#     hours_with_known_prices = 24 - current_hour if current_hour < 13 else (24 - current_hour) + 24

#     # Create forecast
#     if hours_with_known_prices > H:
#         spot_price_forecast = spot_price_data[sim_start_time + t: sim_start_time + t + H].values
#     else:
#         known_prices = spot_price_data[sim_start_time + t: sim_start_time + t + hours_with_known_prices].values
#         baseline_prices = spot_price_baseline[sim_start_time + t + hours_with_known_prices: sim_start_time + t + H]
#         spot_price_forecast = np.concatenate((known_prices, baseline_prices))

#     return spot_price_forecast

# def make_spot_price_forecast(spot_price_data, datetime_index, sim_start_time, t, H):
#     current_hour = datetime_index[t].hour

#     # Determine hours with known prices
#     hours_with_known_prices = min(24 - current_hour if current_hour < 13 else (24 - current_hour) + 24, H)

#     # Known prices
#     known_prices = spot_price_data[sim_start_time + t: sim_start_time + t + hours_with_known_prices].values

#     # If the horizon extends beyond known prices, repeat the last known price
#     if hours_with_known_prices < H:
#         repeated_last_price = np.repeat(known_prices[-1], H - hours_with_known_prices)
#         known_prices = np.concatenate((known_prices, repeated_last_price))

#     return known_prices


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


# def optimize(load, tou_prices, spot_prices, T, datetime_index, Q=20, C=10, D=10, q_init=10, q_final=10, p_prev=[], datetime_index_prev=None, N=1):
#     # Define constants
#     P = 20
#     eff_s, eff_c, eff_d = 0.99998, 0.95, 0.95
#     tier_costs = np.array([83, 147, 252, 371, 490])
#     tier_thresholds = np.array([2, 5, 10, 15])
#     z_max = 20
    
#     # Define variables
#     p, c, d, q = cp.Variable(T), cp.Variable(T), cp.Variable(T), cp.Variable(T+1)

#     # Set up energy usage constraints and cost
#     cons = [0 <= p, p <= P,
#             load + c == p + d,
#             q[1:] == eff_s * q[:-1] + eff_c * c - (1/eff_d) * d,
#             q[0] == q_init, q[-1] == q_final,
#             0 <= q, q <= Q,
#             0 <= c, c <= C,
#             0 <= d, d <= D]
#     energy_cost = cp.sum(cp.multiply(tou_prices + spot_prices, p))

#     # Set up peak power usage constraints and cost
#     z_values = compute_z(p, datetime_index, p_prev, datetime_index_prev, N)
#     peak_power_cost = 0
#     for z in z_values:
#         y = cp.Variable(len(tier_costs), boolean=True)
#         peak_power_cost += cp.matmul(tier_costs, y)
#         cons += [z <= cp.matmul(tier_thresholds, y[:-1]) + z_max * y[-1], cp.sum(y) == 1]

#     # Solve the optimization problem
#     cost = energy_cost + peak_power_cost
#     problem = cp.Problem(cp.Minimize(cost), cons)
#     problem.solve(solver=cp.GUROBI)
    
#     return p.value, q.value, c.value, d.value, cost.value

def optimize(load, tou_prices, spot_prices, T, datetime_index, Q=40, C=20, D=20, q_init=20, q_final=20, p_prev=[], datetime_index_prev=None, N=1, relax=False, resolve=False, y_new=None):
    # Define constants
    P = 20
    eff_s, eff_c, eff_d = 0.99998, 0.95, 0.95
    tier_costs = np.array([83, 147, 252, 371, 490])
    tier_thresholds = np.array([2, 5, 10, 15])
    z_max = 20
    
    # Define variables
    p, c, d, q = cp.Variable(T), cp.Variable(T), cp.Variable(T), cp.Variable(T+1)

    # Set up energy usage constraints and cost
    cons = [0 <= p, p <= P,
            load + c == p + d,
            q[1:] == eff_s * q[:-1] + eff_c * c - (1/eff_d) * d,
            q[0] == q_init, q[-1] == q_final,
            0 <= q, q <= Q,
            0 <= c, c <= C,
            0 <= d, d <= D]
    energy_cost = cp.sum(cp.multiply(tou_prices + spot_prices, p))

    # Set up peak power usage constraints and cost
    z_values = compute_z(p, datetime_index, p_prev, datetime_index_prev, N)
    peak_power_cost = 0
    for i, z in enumerate(z_values):
        if relax:
            y = cp.Variable(len(tier_costs))
            cons += [z <= cp.matmul(tier_thresholds, y[:-1]) + z_max * y[-1], cp.sum(y) == 1, y <= 1, y >=0]
        elif resolve:
            y = y_new[i, :]
            cons += [z <= cp.matmul(tier_thresholds, y[:-1]) + z_max * y[-1]]
        else:   
            y = cp.Variable(len(tier_costs), boolean=True)
            cons += [z <= cp.matmul(tier_thresholds, y[:-1]) + z_max * y[-1], cp.sum(y) == 1]
        peak_power_cost += cp.matmul(tier_costs, y)

    # Solve the optimization problem
    cost = energy_cost + peak_power_cost
    problem = cp.Problem(cp.Minimize(cost), cons)
    if relax or resolve:
        problem.solve()
    else:
        problem.solve(solver=cp.GUROBI)
    
    return p.value, q.value, c.value, d.value, cost.value, problem.status


# def optimize(load, tou_prices, spot_prices, T, datetime_index, K=1, Q=40, C=20, D=20, q_init=20, q_final=20, p_prev=[], datetime_index_prev=None, N=1, relax=False, resolve=False, y_new=None):
#     # Define constants
#     P = 20
#     eff_s, eff_c, eff_d = 0.99998, 0.95, 0.95
#     tier_costs = np.array([83, 147, 252, 371, 490])
#     tier_thresholds = np.array([2, 5, 10, 15, 20])
    
#     # Define variables
#     p = cp.Variable((T, K))
#     c = cp.Variable((T, K))
#     d = cp.Variable((T, K))
#     q = cp.Variable((T+1, K))
    
#     # Set up energy usage constraints and cost
#     cons = [0 <= p, p <= P,
#             load + c == p + d,
#             q[1:, :] == eff_s * q[:-1, :] + eff_c * c - (1/eff_d) * d,
#             q[0, :] == q_init, q[-1, :] == q_final,
#             0 <= q, q <= Q,
#             0 <= c, c <= C,
#             0 <= d, d <= D]
    
#     # All scenarios have the same decision variables in the first time period
#     cons += [p[0, :] - p[0, 0] == 0, c[0, :] - c[0, 0] == 0, d[0, :] - d[0, 0] == 0]
    
#     # Energy cost
#     energy_cost = 0
#     for k in range(K):
#         energy_cost += cp.sum(cp.multiply(tou_prices + spot_prices, p[:, k]))
    
#     # Set up peak power usage constraints and cost
#     peak_power_cost = []
#     for k in range(K):
#         _peak_power_cost = 0
#         z_values = compute_z(p[:, k], datetime_index, p_prev, datetime_index_prev, N)
#         for z in z_values:  
#             y = cp.Variable(len(tier_costs), boolean=True)
#             cons += [z <= cp.matmul(tier_thresholds, y), cp.sum(y) == 1]
#             _peak_power_cost += cp.matmul(tier_costs, y)
#         peak_power_cost.append(_peak_power_cost)

#     peak_power_cost = cp.sum(cp.hstack(peak_power_cost))
#     # Solve the optimization problem
#     cost = (1/K) * (energy_cost + peak_power_cost)
#     problem = cp.Problem(cp.Minimize(cost), cons)
#     problem.solve(verbose=True)
    
#     return p.value, q.value, c.value, d.value, cost.value, problem.status


def shift_one(y):
    for j in range(y.shape[1] - 1):  # Loop over columns, except the last one
        for i in range(y.shape[0]):  # Loop over rows
            if y[i, j] == 1:
                y[i, j] = 0
                y[i, j + 1] = 1
                return y  # Exit function once the first 1 is shifted
    return y  # If no shift was made, return original array


def relax_and_resolve(load, tou_prices, spot_prices, T, datetime_index, q_init=12.5, p_prev=[], datetime_index_prev=None, N=3):
    # Relax integer constraints and solve relaxed problem
    p, q, c, d, cost, status = optimize(load=load, tou_prices=tou_prices, spot_prices=spot_prices, T=T, datetime_index=datetime_index, 
                                        q_init=q_init, p_prev=p_prev, datetime_index_prev=datetime_index_prev, relax=True)
    
    # Recover the relaxed variables and round them down
    z = get_z_values(p, datetime_index, N)
    z = np.floor(z)
    if p_prev:
        z_curr = np.max(p_prev)
        z = np.maximum(z, z_curr)
    y_new = get_y(z)
    
    # Resolve with new integer variables
    p, q, c, d, cost, status = optimize(load=load, tou_prices=tou_prices, spot_prices=spot_prices, T=T, datetime_index=datetime_index, 
                                        q_init=q_init, p_prev=p_prev, datetime_index_prev=datetime_index_prev, resolve=True, y_new=y_new)

    # If the relaxed problem is infeasible, increase the lowest tier and try again
    while status != "optimal":
        y_new = shift_one(y_new)
        p, q, c, d, cost, status = optimize(load=load, tou_prices=tou_prices, spot_prices=spot_prices, T=T, datetime_index=datetime_index, 
                                            q_init=q_init, p_prev=p_prev, datetime_index_prev=datetime_index_prev, resolve=True, y_new=y_new)
    
    return p, q, c, d, cost, status