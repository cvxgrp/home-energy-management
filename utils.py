import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import warnings
import matplotlib
from math import sqrt

def get_month_slices(datetime_index: pd.DatetimeIndex) -> list:
    """
    Creates a list of slices, each representing a continuous range of months within a given DatetimeIndex.

    This function is designed to split a pandas DatetimeIndex into continuous ranges of months. It iterates through the 
    DatetimeIndex, and whenever a change in the month is detected, it creates a slice object representing the range 
    of the previous month and appends it to a list. The final list of slices is then returned.

    Args:
        datetime_index (pd.DatetimeIndex): A pandas DatetimeIndex object containing datetime entries, assumed to be 
                                           sorted in ascending order.

    Returns:
        list: A list of slice objects, where each slice represents a continuous range of months within the input 
              DatetimeIndex.

    Example:
        >>> datetime_index = pd.date_range("2022-01-01", "2022-03-31", freq="D")
        >>> month_slices = create_month_slices(datetime_index)
        >>> print(month_slices)
        [slice(0, 31, None), slice(31, 59, None), slice(59, 90, None)]
    """
    month_slices = []
    start_idx = 0
    curr_month = datetime_index[0].month

    for idx, dt in enumerate(datetime_index[1:], 1):
        if dt.month != curr_month:
            month_slices.append(slice(start_idx, idx))
            start_idx = idx
            curr_month = dt.month

    # Append the slice for the last month
    month_slices.append(slice(start_idx, len(datetime_index)))

    return month_slices

def format_x_axis(ax, display_labels=True):
    # Set x-axis major tick locator and formatter for months
    months = mdates.MonthLocator(interval=1)
    months_fmt = mdates.DateFormatter("%b")
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(months_fmt)
    
    # set the tick labels to be between the ticks
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ticklabels = ax.get_xticklabels()
        ax.set_xticklabels([''] * len(ticklabels))  # hide the original tick labels
        
    # Calculate positions for new labels between ticks
    tick_positions = ax.get_xticks()
    label_positions = (tick_positions[:-1] + tick_positions[1:]) / 2
    label_positions = list(label_positions) + [tick_positions[-1]]
    
    # Set positions of new labels and display them
    ax.set_xticks(label_positions, minor=True)
    if display_labels:
        ax.set_xticklabels(months_fmt.format_ticks(tick_positions), minor=True)
    else:
        ax.set_xticklabels([''] * len(tick_positions), minor=True)
    ax.tick_params(axis='x', which='minor', length=0)
    

def print_costs(cost_usage, cost_peak):
    # Calculate total cost and print the cost results
    total_cost = cost_usage + cost_peak
    print(f"Usage cost: {cost_usage/1e3:,.2f} kNOK ({100 * cost_usage / total_cost:.2f}%)")
    print(f"Peak cost: {cost_peak/1e3:,.2f} kNOK ({100 * cost_peak / total_cost:.2f}%)")
    print(f"Total cost: {total_cost/1e3:,.2f} kNOK") 
    
def plot_power_grid(datetime_index, p_grid, Q):
    p_grid_series = pd.Series(data=p_grid, index=datetime_index)
    p_peak_hourly = p_grid_series.groupby(p_grid_series.index.to_period('M')).transform('max').to_numpy()

    # Create plot of power pulled from the grid and peak demand
    fig, ax = plt.subplots()
    ax.plot(datetime_index, p_grid, color="blue", label="Power pulled from the grid")
    ax.plot(datetime_index, p_peak_hourly, color="red", label="Monthly peak demand")
    ax.set_ylabel("Power (kW)")
    ax.grid(True)
    
    # Set plot title and legend
    ax.set_title("No storage" if Q == 0 else f"Storage capacity: {Q:.0f} kW")
    format_x_axis(ax)
    if Q == 0:
        ax.legend(loc='lower center', ncol=2)
        
def plot_state_of_charge(datetime_index, q):
    # Create plot of power pulled from the grid and peak demand
    fig, ax = plt.subplots()
    ax.plot(datetime_index, q, color="blue")
    ax.set_ylabel("State of charge (kWh)")
    ax.grid(True)
    format_x_axis(ax)

def latexify(fig_width=None, fig_height=None, columns=1):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert(columns in [1,2])

    if fig_width is None:
        fig_width = 3.39 if columns==1 else 6.9 # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width*golden_mean # Height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height + 
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES
    
    params = {'backend': 'ps',
              'text.latex.preamble': ['\\usepackage{gensymb}'],
              'axes.labelsize': 12,
              'axes.titlesize': 12,
              'xtick.labelsize': 12,
              'ytick.labelsize': 12,
              'legend.fontsize': 12,
              'text.usetex': True,
              'figure.figsize': [fig_width,fig_height],
              'font.family': 'serif',
              'font.serif': 'Computer Modern Roman',
              'savefig.dpi': 200,
              'savefig.format': 'pdf',
              'savefig.bbox': 'tight'
    }

    matplotlib.rcParams.update(params)