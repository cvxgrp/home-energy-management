# import pandas as pd
# import matplotlib.dates as mdates
# import matplotlib.pyplot as plt
# import warnings
# import matplotlib
# from math import sqrt

# def format_x_axis(ax, display_labels=True):
#     # Set x-axis major tick locator and formatter for months
#     months = mdates.MonthLocator(interval=1)
#     months_fmt = mdates.DateFormatter("%b")
#     ax.xaxis.set_major_locator(months)
#     ax.xaxis.set_major_formatter(months_fmt)
    
#     # set the tick labels to be between the ticks
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         ticklabels = ax.get_xticklabels()
#         ax.set_xticklabels([''] * len(ticklabels))  # hide the original tick labels
        
#     # Calculate positions for new labels between ticks
#     tick_positions = ax.get_xticks()
#     label_positions = (tick_positions[:-1] + tick_positions[1:]) / 2
#     label_positions = list(label_positions) + [tick_positions[-1]]
    
#     # Set positions of new labels and display them
#     ax.set_xticks(label_positions, minor=True)
#     if display_labels:
#         ax.set_xticklabels(months_fmt.format_ticks(tick_positions), minor=True)
#     else:
#         ax.set_xticklabels([''] * len(tick_positions), minor=True)
#     ax.tick_params(axis='x', which='minor', length=0)

# def latexify(fig_width=None, fig_height=None, columns=1):
#     """Set up matplotlib's RC params for LaTeX plotting.
#     Call this before plotting a figure.

#     Parameters
#     ----------
#     fig_width : float, optional, inches
#     fig_height : float,  optional, inches
#     columns : {1, 2}
#     """

#     # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

#     # Width and max height in inches for IEEE journals taken from
#     # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

#     assert(columns in [1,2])

#     if fig_width is None:
#         fig_width = 3.39 if columns==1 else 6.9 # width in inches

#     if fig_height is None:
#         golden_mean = (sqrt(5)-1.0)/2.0    # Aesthetic ratio
#         fig_height = fig_width*golden_mean # Height in inches

#     MAX_HEIGHT_INCHES = 8.0
#     if fig_height > MAX_HEIGHT_INCHES:
#         print("WARNING: fig_height too large:" + fig_height + 
#               "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
#         fig_height = MAX_HEIGHT_INCHES
    
#     params = {'backend': 'ps',
#               'text.latex.preamble': ['\\usepackage{gensymb}'],
#               'axes.labelsize': 11,
#               'axes.titlesize': 11,
#               'xtick.labelsize': 11,
#               'ytick.labelsize': 11,
#               'legend.fontsize': 11,
#               'text.usetex': True,
#               'figure.figsize': [fig_width,fig_height],
#               'font.family': 'serif',
#               'font.serif': 'Computer Modern Roman',
#               'savefig.dpi': 200,
#               'savefig.format': 'pdf',
#               'savefig.bbox': 'tight',
#               'lines.linewidth': 1.5
#     }

#     matplotlib.rcParams.update(params)

import matplotlib.pyplot as plt
from math import sqrt

def latexify(fig_width=None, fig_height=None, columns=1, font_size=12, line_width=1.5, markersize=4):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float, optional, inches
    columns : {1, 2}
    font_size : int, optional
    line_width : float, optional
    markersize : int, optional
    """

    assert(columns in [1, 2])

    if fig_width is None:
        fig_width = 3.39 if columns == 1 else 6.9  # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
        fig_height = fig_width * golden_mean  # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + str(fig_height) +
              "so will reduce to" + str(MAX_HEIGHT_INCHES) + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {'backend': 'ps',
              'text.latex.preamble': [r'\usepackage{gensymb}', r'\usepackage{amsmath}'],
              'axes.labelsize': font_size,
              'axes.titlesize': font_size,
              'xtick.labelsize': font_size,
              'ytick.labelsize': font_size,
              'legend.fontsize': font_size,
              'text.usetex': True,
              'figure.figsize': [fig_width, fig_height],
              'font.family': 'serif',
              'font.serif': 'Computer Modern Roman',
              'savefig.dpi': 200,
              'savefig.format': 'pdf',
              'savefig.bbox': 'tight',
              'lines.linewidth': line_width,
              'lines.markersize': markersize,
              'axes.grid': True,
              'grid.alpha': 0.53
    }

    plt.rcParams.update(params)

