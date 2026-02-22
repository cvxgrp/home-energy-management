"""Reproduce all results and figures for the paper.

Usage:
    uv run python run.py                    # Run everything
    uv run python run.py --train            # Train forecasters only
    uv run python run.py --experiments      # Run experiments only
    uv run python run.py --figures          # Generate figures only
    uv run python run.py --baseline         # Run baseline policies
    uv run python run.py --prescient        # Run prescient optimization
    uv run python run.py --mpc              # Run MPC policies
    uv run python run.py --capacity         # Run capacity sweep
    uv run python run.py --sensitivity      # Run sensitivity analysis
"""

import argparse
import datetime
import pickle
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd
from tqdm import tqdm

from hem.data import load_data, load_forecaster, get_eval_window
from hem.forecast import (
    train_baseline,
    train_ar_model,
    featurize_baseline,
    AR_LOOKBACK,
    AR_HORIZON,
)
from hem.policies import (
    no_storage,
    peak_shaving,
    energy_arbitrage,
    prescient,
    mpc,
    optimize,
    compute_costs,
    get_z_values,
    TIER_THRESHOLDS,
)
from hem.plot import (
    latexify,
    plot_cost_vs_capacity,
    plot_z_comparison,
    plot_grid_power,
    plot_load_year,
    plot_charge_level_year,
    plot_week_load,
    plot_week_grid_power,
    plot_week_soc,
    plot_multiyear,
    plot_one_week,
    plot_baseline_comparison,
    plot_forecast_comparison,
)


DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")


def save_results(name: str, data: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"{name}.pkl", "wb") as f:
        pickle.dump(data, f)


def load_results(name: str) -> dict:
    path = RESULTS_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run experiments first.")
    with open(path, "rb") as f:
        return pickle.load(f)


def train_forecasters():
    print("Training forecasters...")

    load_data = pd.read_csv(DATA_DIR / "loads.csv", parse_dates=[0], index_col=0)[
        "Load (kW)"
    ]
    da_price_data = pd.read_csv(
        DATA_DIR / "da_prices.csv", parse_dates=[0], index_col=0
    )["DA Price (NOK/kWh)"]

    load_train = load_data[load_data.index.year <= 2021]
    price_train = da_price_data[da_price_data.index.year <= 2021]

    load_theta = train_baseline(load_train, tau=0.2, lambd=0.1)
    price_theta = train_baseline(price_train, tau=0.5, lambd=0.1)

    train_len = len(load_train)
    X_train = np.array([featurize_baseline(t) for t in range(train_len)])
    train_baseline_load = pd.Series(X_train @ load_theta, index=load_train.index)

    date_range = pd.date_range(
        start="2022-01-01 00:00:00", end="2023-12-31 23:00:00", freq="h"
    )
    X_ext = np.array(
        [featurize_baseline(train_len + i) for i in range(len(date_range))]
    )
    ext_baseline = pd.Series(X_ext @ load_theta, index=date_range)
    load_baseline = pd.concat([train_baseline_load, ext_baseline])
    load_baseline.to_csv(DATA_DIR / "load_baseline.csv", header=["Baseline load (kW)"])

    X_train_price = np.array([featurize_baseline(t) for t in range(len(price_train))])
    train_baseline_price = pd.Series(
        X_train_price @ price_theta, index=price_train.index
    )
    X_ext_price = np.array(
        [featurize_baseline(len(price_train) + i) for i in range(len(date_range))]
    )
    ext_baseline_price = pd.Series(X_ext_price @ price_theta, index=date_range)
    price_baseline = pd.concat([train_baseline_price, ext_baseline_price])
    price_baseline.to_csv(
        DATA_DIR / "da_price_baseline.csv",
        header=["Baseline day-ahead price (NOK/kWh)"],
    )

    load_residuals = (load_train - train_baseline_load).values
    load_ar_params = train_ar_model(
        load_residuals, AR_LOOKBACK, AR_HORIZON, tau=0.2, lambd=0.1
    )
    pd.to_pickle(load_ar_params, DATA_DIR / "load_AR_params.pkl")

    price_residuals = (price_train - train_baseline_price).values
    price_ar_params = train_ar_model(
        price_residuals, AR_LOOKBACK, AR_HORIZON, tau=0.5, lambd=0.1
    )
    pd.to_pickle(price_ar_params, DATA_DIR / "da_price_AR_params.pkl")


def run_baseline(sim: dict) -> None:
    print("Running baseline policies...")
    results = {}

    for name, policy in [
        ("no_storage", no_storage),
        ("peak_shaving", peak_shaving),
        ("energy_arbitrage", energy_arbitrage),
    ]:
        flows = policy(sim)
        costs = compute_costs(
            sim["tou_prices"], sim["da_prices"], flows["p"], sim["datetime_index"]
        )
        results[name] = {"flows": flows, "costs": costs}
        print(f"  {name}: {costs['total']:.0f} NOK")

    save_results("baseline", results)


def run_prescient(sim: dict) -> None:
    print("Running prescient...")
    flows = prescient(sim, solver=cp.GUROBI)
    costs = compute_costs(
        sim["tou_prices"], sim["da_prices"], flows["p"], sim["datetime_index"]
    )
    results = {"flows": flows, "costs": costs}
    print(f"  prescient: {costs['total']:.0f} NOK")

    save_results("prescient", results)


def run_mpc(sim: dict, data: dict, forecaster: dict) -> None:
    print("Running MPC...")
    results = {}

    flows = mpc(sim, data, forecaster, solver=cp.GUROBI)
    costs = compute_costs(
        sim["tou_prices"], sim["da_prices"], flows["p"], sim["datetime_index"]
    )
    results["mpc"] = {"flows": flows, "costs": costs}
    print(f"  mpc: {costs['total']:.0f} NOK")

    flows = mpc(sim, data, forecaster, include_peak_power=False, solver=cp.MOSEK)
    costs = compute_costs(
        sim["tou_prices"], sim["da_prices"], flows["p"], sim["datetime_index"]
    )
    results["mpc_energy_only"] = {"flows": flows, "costs": costs}
    print(f"  mpc_energy_only: {costs['total']:.0f} NOK")

    save_results("mpc", results)


def run_capacity_sweep(data: dict) -> None:
    print("Running capacity sweep...")
    capacities = np.linspace(0, 100, 51)
    results = {"capacities": capacities}

    for year in [2020, 2021, 2022]:
        sim = get_eval_window(data, year=year)
        costs = np.zeros(len(capacities))

        for i, cap in enumerate(tqdm(capacities, desc="Capacity sweep")):
            result = optimize(
                load=sim["load"],
                tou_prices=sim["tou_prices"],
                da_prices=sim["da_prices"],
                datetime_index=sim["datetime_index"],
                capacity=cap,
                max_charge=cap / 2,
                max_discharge=cap / 2,
                q_init=cap / 2,
                q_final=cap / 2,
                include_peak_power=True,
                peak_power_N=3,
                solver=cp.GUROBI,
            )
            costs[i] = result["cost"]

        results[str(year)] = costs

    save_results("capacity_sweep", results)


def save_sim_data(sim: dict) -> None:
    data = {
        "datetime_index": sim["datetime_index"],
        "load": sim["load"],
        "tou_prices": sim["tou_prices"],
        "da_prices": sim["da_prices"],
    }
    save_results("sim_data", data)


def generate_data_figures(data: dict):

    load = data["load"]
    da_prices = data["da_prices"]

    load_plot = load[load.index.year <= 2022]
    da_plot = da_prices[da_prices.index.year <= 2022]

    plot_multiyear(load_plot, "Load (kW)", FIGURES_DIR / "load_3y.pdf")
    plot_one_week(
        load, "2022-01-10", "2022-01-16", "Load (kW)", FIGURES_DIR / "load_week.pdf"
    )
    plot_multiyear(
        da_plot, "Day-ahead price (NOK/kWh)", FIGURES_DIR / "da_prices_3y.pdf"
    )
    plot_one_week(
        da_prices,
        "2022-01-03",
        "2022-01-09",
        "Day-ahead price (NOK/kWh)",
        FIGURES_DIR / "da_prices_week.pdf",
    )


def generate_forecast_figures(data: dict, forecaster: dict):

    load = data["load"]
    da_prices = data["da_prices"]
    load_baseline = forecaster["load_baseline"]
    da_price_baseline = forecaster["da_price_baseline"]
    load_ar_params = forecaster["load_ar_params"]
    da_price_ar_params = forecaster["da_price_ar_params"]

    test_load = load[load.index.year == 2022]
    test_baseline_load = load_baseline[load_baseline.index.year == 2022]
    test_prices = da_prices[da_prices.index.year == 2022]
    test_baseline_price = da_price_baseline[da_price_baseline.index.year == 2022]

    plot_baseline_comparison(
        test_load,
        test_baseline_load,
        "2022-01-03",
        "2022-01-09",
        "Load (kW)",
        FIGURES_DIR / "load_baseline_jan.pdf",
    )
    plot_baseline_comparison(
        test_load,
        test_baseline_load,
        "2022-06-06",
        "2022-06-12",
        "Load (kW)",
        FIGURES_DIR / "load_baseline_jun.pdf",
    )
    plot_baseline_comparison(
        test_prices,
        test_baseline_price,
        "2022-01-10",
        "2022-01-16",
        "DA price (NOK/kWh)",
        FIGURES_DIR / "price_baseline_jan.pdf",
    )
    plot_baseline_comparison(
        test_prices,
        test_baseline_price,
        "2022-06-13",
        "2022-06-19",
        "DA price (NOK/kWh)",
        FIGURES_DIR / "price_baseline_jun.pdf",
    )

    M, L = AR_LOOKBACK, AR_HORIZON
    test_day = pd.Timestamp(datetime.datetime(2022, 1, 5, 12, 0))
    past = test_load[test_load.index <= test_day].iloc[-M:]
    past_baseline = test_baseline_load[test_baseline_load.index <= test_day].iloc[-M:]
    future = test_load[test_load.index > test_day].iloc[:L]
    future_baseline = test_baseline_load[test_baseline_load.index > test_day].iloc[:L]

    past_res = past - past_baseline
    future_res_pred = past_res.values @ load_ar_params

    plot_forecast_comparison(
        past.values,
        future.values,
        future_baseline.values,
        future_baseline.values + future_res_pred,
        past.index,
        future.index,
        "Load (kW)",
        FIGURES_DIR / "load_forecast.pdf",
    )

    test_day_price = pd.Timestamp(datetime.datetime(2022, 5, 12, 12, 0))
    past_p = test_prices[test_prices.index <= test_day_price].iloc[-M:]
    past_baseline_p = test_baseline_price[
        test_baseline_price.index <= test_day_price
    ].iloc[-M:]
    future_p = test_prices[test_prices.index > test_day_price].iloc[:L]
    future_baseline_p = test_baseline_price[
        test_baseline_price.index > test_day_price
    ].iloc[:L]

    past_res_p = past_p - past_baseline_p
    future_res_pred_p = past_res_p.values @ da_price_ar_params
    future_ar_p = future_baseline_p.values + future_res_pred_p

    current_hour = test_day_price.hour
    hours_known = min(
        24 - current_hour if current_hour < 13 else (24 - current_hour) + 24, L
    )
    future_baseline_adj = future_baseline_p.copy()
    future_ar_adj = future_ar_p.copy()
    future_baseline_adj.iloc[:hours_known] = future_p.iloc[:hours_known]
    future_ar_adj[:hours_known] = future_p.iloc[:hours_known].values

    plot_forecast_comparison(
        past_p.values,
        future_p.values,
        future_baseline_adj.values,
        future_ar_adj,
        past_p.index,
        future_p.index,
        "Day-ahead price (NOK/kWh)",
        FIGURES_DIR / "price_forecast.pdf",
    )


def generate_policy_figures(sim_data: dict, prescient_results: dict, mpc_results: dict):

    datetime_index = sim_data["datetime_index"]
    load = sim_data["load"]

    z_no_storage = get_z_values(load, datetime_index, N=3)
    tier_lines = [2, 5, 10]

    plot_load_year(load, datetime_index, FIGURES_DIR / "load_year.pdf")

    week_start = datetime_index.get_loc(pd.Timestamp("2022-07-04"))
    week_end = week_start + 24 * 7
    load_week = load[week_start:week_end]

    tou_prices = sim_data["tou_prices"]
    da_prices = sim_data["da_prices"]
    prices_week = tou_prices[week_start:week_end] + da_prices[week_start:week_end]

    capacity = 40.0

    p_prescient = prescient_results["flows"]["p"]
    q_prescient = prescient_results["flows"]["q"]
    z_prescient = get_z_values(p_prescient, datetime_index, N=3)

    p_mpc = mpc_results["mpc"]["flows"]["p"]
    q_mpc = mpc_results["mpc"]["flows"]["q"]
    z_mpc = get_z_values(p_mpc, datetime_index, N=3)

    all_power_week = np.concatenate(
        [load_week, p_prescient[week_start:week_end], p_mpc[week_start:week_end]]
    )
    power_ylim = (min(0, all_power_week.min()), all_power_week.max() * 1.05)

    plot_grid_power(
        p_prescient, datetime_index, tier_lines, FIGURES_DIR / "power_prescient.pdf"
    )
    plot_charge_level_year(
        q_prescient,
        datetime_index,
        capacity,
        FIGURES_DIR / "charge_level_prescient.pdf",
    )
    plot_z_comparison(
        z_no_storage,
        z_prescient,
        TIER_THRESHOLDS,
        ("No storage", "Prescient"),
        FIGURES_DIR / "peak_avg_prescient.pdf",
    )

    plot_week_load(
        load_week, prices_week, FIGURES_DIR / "week_load_prescient.pdf", ylim=power_ylim
    )
    plot_week_grid_power(
        p_prescient[week_start:week_end],
        prices_week,
        FIGURES_DIR / "week_power_prescient.pdf",
        ylim=power_ylim,
    )
    plot_week_soc(
        q_prescient[week_start : week_end + 1],
        prices_week,
        FIGURES_DIR / "week_soc_prescient.pdf",
    )

    plot_grid_power(p_mpc, datetime_index, tier_lines, FIGURES_DIR / "power_mpc.pdf")
    plot_charge_level_year(
        q_mpc, datetime_index, capacity, FIGURES_DIR / "charge_level_mpc.pdf"
    )
    plot_z_comparison(
        z_no_storage,
        z_mpc,
        TIER_THRESHOLDS,
        ("No storage", "MPC"),
        FIGURES_DIR / "peak_avg_mpc.pdf",
    )

    plot_week_load(
        load_week, prices_week, FIGURES_DIR / "week_load_mpc.pdf", ylim=power_ylim
    )
    plot_week_grid_power(
        p_mpc[week_start:week_end],
        prices_week,
        FIGURES_DIR / "week_power_mpc.pdf",
        ylim=power_ylim,
    )
    plot_week_soc(
        q_mpc[week_start : week_end + 1], prices_week, FIGURES_DIR / "week_soc_mpc.pdf"
    )


def generate_capacity_figure(capacity_sweep: dict):

    capacities = capacity_sweep["capacities"]
    savings_dict = {}

    for year in ["2020", "2021", "2022"]:
        costs = capacity_sweep[year]
        baseline_cost = costs[0]
        savings_pct = 100 * (baseline_cost - costs) / baseline_cost
        savings_dict[year] = savings_pct

    plot_cost_vs_capacity(
        capacities, savings_dict, FIGURES_DIR / "savings_vs_capacity.pdf"
    )


def run_sensitivity(sim: dict, data: dict, forecaster: dict) -> None:
    print("Running sensitivity analysis...")
    results = {}

    # Horizon sweep
    horizons = [24, 168, 360, 720, 1440]
    horizon_costs = {}
    for H in horizons:
        flows = mpc(sim, data, forecaster, horizon=H, solver=cp.GUROBI)
        costs = compute_costs(
            sim["tou_prices"], sim["da_prices"], flows["p"], sim["datetime_index"]
        )
        horizon_costs[H] = costs
    results["horizon"] = horizon_costs

    # Peak power N sweep
    n_values = [1, 3]
    n_costs = {}
    for N in n_values:
        flows = mpc(sim, data, forecaster, peak_power_N=N, solver=cp.GUROBI)
        costs = compute_costs(
            sim["tou_prices"], sim["da_prices"], flows["p"], sim["datetime_index"]
        )
        n_costs[N] = costs
    results["peak_power_N"] = n_costs

    # Forecast method: baseline-only vs baseline+AR
    forecaster_baseline_only = {
        **forecaster,
        "load_ar_params": np.zeros_like(forecaster["load_ar_params"]),
        "da_price_ar_params": np.zeros_like(forecaster["da_price_ar_params"]),
    }
    forecast_costs = {}

    flows = mpc(sim, data, forecaster, solver=cp.GUROBI)
    costs = compute_costs(
        sim["tou_prices"], sim["da_prices"], flows["p"], sim["datetime_index"]
    )
    forecast_costs["baseline_ar"] = costs

    flows = mpc(sim, data, forecaster_baseline_only, solver=cp.GUROBI)
    costs = compute_costs(
        sim["tou_prices"], sim["da_prices"], flows["p"], sim["datetime_index"]
    )
    forecast_costs["baseline_only"] = costs

    results["forecast"] = forecast_costs

    save_results("sensitivity", results)
    print_sensitivity(results)


def print_sensitivity(results: dict) -> None:
    print("\nHorizon sweep:")
    print(f"{'H (hours)':>12}  {'Energy':>10}  {'Peak':>10}  {'Total':>10}")
    for H, costs in results["horizon"].items():
        print(
            f"{H:>12}  {costs['tou'] + costs['da']:>10.0f}"
            f"  {costs['peak']:>10.0f}  {costs['total']:>10.0f}"
        )

    print("\nPeak power N:")
    print(f"{'N':>12}  {'Energy':>10}  {'Peak':>10}  {'Total':>10}")
    for N, costs in results["peak_power_N"].items():
        print(
            f"{N:>12}  {costs['tou'] + costs['da']:>10.0f}"
            f"  {costs['peak']:>10.0f}  {costs['total']:>10.0f}"
        )

    print("\nForecast method:")
    print(f"{'Method':>16}  {'Energy':>10}  {'Peak':>10}  {'Total':>10}")
    for method, costs in results["forecast"].items():
        print(
            f"{method:>16}  {costs['tou'] + costs['da']:>10.0f}"
            f"  {costs['peak']:>10.0f}  {costs['total']:>10.0f}"
        )


def generate_figures():
    print("Generating figures...")
    FIGURES_DIR.mkdir(exist_ok=True)
    latexify(fig_width=5)

    data = load_data()
    forecaster = load_forecaster()

    sim_data = load_results("sim_data")
    prescient_results = load_results("prescient")
    mpc_results = load_results("mpc")
    capacity_sweep = load_results("capacity_sweep")

    generate_data_figures(data)
    generate_forecast_figures(data, forecaster)
    generate_policy_figures(sim_data, prescient_results, mpc_results)
    generate_capacity_figure(capacity_sweep)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--train", action="store_true", help="Train forecasters")
    parser.add_argument(
        "--experiments", action="store_true", help="Run all experiments"
    )
    parser.add_argument("--figures", action="store_true", help="Generate figures")
    parser.add_argument("--baseline", action="store_true", help="Run baseline policies")
    parser.add_argument(
        "--prescient", action="store_true", help="Run prescient optimization"
    )
    parser.add_argument("--mpc", action="store_true", help="Run MPC policies")
    parser.add_argument("--capacity", action="store_true", help="Run capacity sweep")
    parser.add_argument(
        "--sensitivity", action="store_true", help="Run sensitivity analysis"
    )
    args = parser.parse_args()

    run_all = not any(
        [
            args.train,
            args.experiments,
            args.figures,
            args.baseline,
            args.prescient,
            args.mpc,
            args.capacity,
            args.sensitivity,
        ]
    )

    if run_all or args.train:
        train_forecasters()

    if (
        run_all
        or args.experiments
        or args.baseline
        or args.prescient
        or args.mpc
        or args.capacity
        or args.sensitivity
    ):
        data = load_data()
        forecaster = load_forecaster()
        sim = get_eval_window(data, year=2022)
        save_sim_data(sim)

        if run_all or args.experiments or args.baseline:
            run_baseline(sim)

        if run_all or args.experiments or args.prescient:
            run_prescient(sim)

        if run_all or args.experiments or args.mpc:
            run_mpc(sim, data, forecaster)

        if run_all or args.experiments or args.capacity:
            run_capacity_sweep(data)

        if args.sensitivity:
            run_sensitivity(sim, data, forecaster)

    if run_all or args.figures:
        generate_figures()


if __name__ == "__main__":
    main()
