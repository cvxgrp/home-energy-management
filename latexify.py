import matplotlib.pyplot as plt
from math import sqrt

def latexify(fig_width=None, fig_height=None, columns=1, font_size=10):
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
        fig_height = fig_width*golden_mean  # height in inches

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
              'figure.figsize': [fig_width, fig_height],
              'text.usetex': True,
              'font.family': 'serif',
              'font.serif': 'Computer Modern Roman',
              'savefig.dpi': 200,
              'savefig.format': 'pdf',
              'savefig.bbox': 'tight',
              'axes.grid': True,
              'grid.alpha': 0.53
    }

    plt.rcParams.update(params)