import numpy as np
import pandas as pd
import calendar
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cvxpy as cp

# def affine_peak_cost_function(z):
#     a = 20.35
#     b = 83
#     return a * z + b 

def tiered_peak_cost_function(z):
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

def get_tier_upper_bounds(z, P=20):
    if 0 <= z <= 2:
        return 2
    elif 2 < z <= 5:
        return 5
    elif 5 < z <= 10:
        return 10
    elif 10 < z <= 15:
        return 15
    else:
        return P

def get_tier(z):
    if 0 <= z <= 2:
        return 0
    elif 2 < z <= 5:
        return 1
    elif 5 < z <= 10:
        return 2
    elif 10 < z <= 15:
        return 3
    else:
        return 4

def compute_monthly_costs(tou_prices, spot_prices, power, datetime_index, peak_cost_function, N=3):
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
        month_tou_cost = peak_cost_function(z)
        monthly_peak_costs.append(month_tou_cost)
        
    return monthly_tou_costs, monthly_spot_costs, monthly_peak_costs


def compute_total_costs(tou_prices, spot_prices, power, datetime_index, peak_cost_function, N=3):
    monthly_tou_costs, monthly_spot_costs, monthly_peak_costs = compute_monthly_costs(tou_prices, spot_prices, power, datetime_index, peak_cost_function, N)
    
    total_tou_cost = sum(monthly_tou_costs)
    total_spot_cost = sum(monthly_spot_costs)
    total_peak_cost = sum(monthly_peak_costs)
    
    return total_tou_cost, total_spot_cost, total_peak_cost

def print_cost_summary(total_tou_cost, total_spot_cost, total_peak_cost):
    total_cost = total_tou_cost + total_spot_cost + total_peak_cost
    print(f"Total cost: {total_cost:,.2f} NOK")
    print(f"\tEnergy cost related to time-of-use prices: {total_tou_cost:,.2f} NOK ({100 * total_tou_cost / total_cost:.2f}% of total cost)")
    print(f"\tEnergy cost related to day-ahead spot prices: {total_spot_cost:,.2f} NOK ({100 * total_spot_cost / total_cost:.2f}% of total cost)")
    print(f"\tPeak power cost: {total_peak_cost:,.2f} NOK ({100 * total_peak_cost / total_cost:.2f}% of total cost)\n")

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

def make_spot_price_forecast(spot_price_data, spot_price_baseline, datetime_index, sim_start_time, t, H):
    current_hour = datetime_index[t].hour

    # Determine hours with known prices
    hours_with_known_prices = 24 - current_hour if current_hour < 13 else (24 - current_hour) + 24

    # Create forecast
    if hours_with_known_prices > H:
        spot_price_forecast = spot_price_data[sim_start_time + t: sim_start_time + t + H].values
    else:
        known_prices = spot_price_data[sim_start_time + t: sim_start_time + t + hours_with_known_prices].values
        baseline_prices = spot_price_baseline[sim_start_time + t + hours_with_known_prices: sim_start_time + t + H]
        spot_price_forecast = np.concatenate((known_prices, baseline_prices))

    return spot_price_forecast

def update_N_largest_daily_powers(N_largest_daily_powers, max_daily_power, N=3):
    if len(N_largest_daily_powers) < N:
        N_largest_daily_powers.append(max_daily_power)
    else:
        min_value = min(N_largest_daily_powers)
        if max_daily_power > min_value:
            N_largest_daily_powers[N_largest_daily_powers.index(min_value)] = max_daily_power
    return N_largest_daily_powers

def create_shared_cost_and_cons(load, tou_prices, spot_prices, T, Q, C, D, q_init, q_final):
    # Constants
    eff_s = 0.9995 # Storage efficiency
    eff_c = 0.95 # Charging efficiency
    eff_d = 0.95 # Discharging efficiency
    P = 20 # Max power
    
    # Variables
    p = cp.Variable(T)  # Grid power
    u_c = cp.Variable(T)  # Charging battery power
    u_d = cp.Variable(T)  # Discharging battery power
    q = cp.Variable(T+1)  # State of charge
    
    # Constraints
    cons = [0 <= p, p <= P,
            load + u_c == p + u_d,
            q[1:] == eff_s * q[:-1] + eff_c * u_c - u_d / eff_d,
            q[0] == q_init, q[-1] == q_final,
            0 <= q, q <= Q,
            0 <= u_c, u_c <= C,
            0 <= u_d, u_d <= D]
    
    energy_cost = cp.sum(cp.multiply(tou_prices + spot_prices, p))
    
    return energy_cost, cons, p, q, u_c, u_d

def compute_month_z_values(p, datetime_index, prev_N_largest_daily_powers=[], curr_max_daily_power=0, N=3):
    unique_months = pd.unique(datetime_index.month)
    z_values = []
    
    for month in unique_months:
        month_mask = datetime_index.month == month
        unique_days_in_month = pd.unique(datetime_index[month_mask].date)

        daily_max_powers = []

        for day in unique_days_in_month:
            day_mask = (datetime_index.date == day) & month_mask
            if curr_max_daily_power != 0 and day == datetime_index[0].day:
                daily_max_powers.append(cp.maximum(cp.max(p[day_mask]), curr_max_daily_power))
            else:
                daily_max_powers.append(cp.max(p[day_mask]))
        
        if month == datetime_index[0].month:
            z = cp.sum_largest(cp.hstack(daily_max_powers + prev_N_largest_daily_powers), N) / N
            z_values.append(z)
        else:
            z = cp.sum_largest(cp.hstack(daily_max_powers), min(N, len(unique_days_in_month))) / min(N, len(unique_days_in_month))
            z_values.append(z)
        
    return z_values