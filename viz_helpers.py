"""
===============================================================================
VISUALIZATION HELPERS
===============================================================================
Shared plotting utilities and IEEE-style configuration for thermal analysis plots.
===============================================================================
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# IEEE plotting configuration for publication-quality figures
IEEE_CONFIG = {
    'figure_width': 7.16,    # IEEE single column width in inches
    'figure_height': 9.0,     # Adjustable height
    'font_size': 8,
    'title_size': 8,
    'label_size': 8,
    'legend_size': 8,
    'line_width': 1.5,
    'marker_size': 4,
    'dpi': 300
}


def setup_ieee_plot_style():
    """Configure matplotlib to use IEEE publication style"""
    plt.rcParams['font.size'] = IEEE_CONFIG['font_size']
    plt.rcParams['axes.labelsize'] = IEEE_CONFIG['label_size']
    plt.rcParams['axes.titlesize'] = IEEE_CONFIG['title_size']
    plt.rcParams['legend.fontsize'] = IEEE_CONFIG['legend_size']
    plt.rcParams['lines.linewidth'] = IEEE_CONFIG['line_width']
    plt.rcParams['lines.markersize'] = IEEE_CONFIG['marker_size']
    plt.rcParams['figure.dpi'] = IEEE_CONFIG['dpi']


def seconds_to_minutes_formatter(x, pos):
    """Format time axis from seconds to minutes"""
    return f'{int(x/60)}'


def get_time_formatter():
    """Get matplotlib formatter for converting seconds to minutes"""
    return FuncFormatter(seconds_to_minutes_formatter)


def create_stats_textbox(temps, ax, position='top_left'):
    """
    Add statistics textbox to plot
    
    Args:
        temps: Array of temperature values
        ax: Matplotlib axis object
        position: 'top_left', 'top_right', 'bottom_left', 'bottom_right'
    
    Returns:
        Text object added to plot
    """
    import numpy as np
    
    if len(temps) == 0:
        return None
    
    stats_text = f"n={len(temps)}\nμ={np.mean(temps):.1f}°C\nσ={np.std(temps):.1f}°C"
    
    # Position mapping
    position_map = {
        'top_left': (0.02, 0.98, 'top'),
        'top_right': (0.98, 0.98, 'top'),
        'bottom_left': (0.02, 0.02, 'bottom'),
        'bottom_right': (0.98, 0.02, 'bottom')
    }
    
    x, y, va = position_map.get(position, (0.02, 0.98, 'top'))
    ha = 'left' if x < 0.5 else 'right'
    
    text_obj = ax.text(x, y, stats_text, transform=ax.transAxes,
                      verticalalignment=va, horizontalalignment=ha,
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    return text_obj


def save_plot(output_path, fmt='png'):
    """
    Save current plot in specified format using plt.savefig
    
    Args:
        output_path: Base output path (without extension)
        fmt: Format to save ('png', 'pdf', etc.)
    
    Returns:
        Path to saved file
    """
    file_path = f"{output_path}.{fmt}"
    
    if fmt == 'png':
        plt.savefig(file_path, dpi=IEEE_CONFIG['dpi'], bbox_inches='tight')
    else:
        plt.savefig(file_path, bbox_inches='tight')
    
    return file_path
