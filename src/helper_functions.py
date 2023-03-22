import pandas as pd
import matplotlib.dates as mdates
import warnings

# Create month_slices function
def create_month_slices(datetime_index: pd.DatetimeIndex) -> list:
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

def format_x_axis(ax):
    # set the x-axis major tick locator and formatter
    months = mdates.MonthLocator(interval=1)
    months_fmt = mdates.DateFormatter("%b")
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(months_fmt)

    # set the tick labels to be between the ticks
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ticklabels = ax.get_xticklabels()
        ax.set_xticklabels([''] * len(ticklabels)) # hide the original tick labels

    tick_positions = ax.get_xticks()
    label_positions = (tick_positions[:-1] + tick_positions[1:]) / 2 # calculate the positions of the new labels
    label_positions = list(label_positions) + [tick_positions[-1]] # add the last tick position to the list
    ax.set_xticks(label_positions, minor=True) # set the positions of the new labels
    ax.set_xticklabels(months_fmt.format_ticks(tick_positions), minor=True) # set the new labels
    ax.tick_params(axis='x', which='minor', length=0) # hide the minor ticks