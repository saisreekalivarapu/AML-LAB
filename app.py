import os
import re
import time
from flask import Flask, render_template, request, redirect, url_for, flash
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = 'aimllab2_secret_key'  # Needed for flash messages

# Ensure the plots directory exists
PLOTS_DIR = os.path.join(app.root_path, 'static', 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

def parse_input(input_str):
    """
    Parses a string of numbers separated by spaces, commas, or newlines.
    Returns a list of floats.
    """
    if not input_str:
        return []
    # Replace common delimiters with spaces, then split
    tokens = re.split(r'[\s,]+', input_str.strip())
    # Filter empty tokens and convert to float
    return [float(t) for t in tokens if t]

@app.route('/')
def index():
    return render_template('index.html', x_input='', y_input='')

@app.route('/calculate', methods=['POST'])
def calculate():
    x_raw = request.form.get('x_input', '')
    y_raw = request.form.get('y_input', '')

    # Validation
    try:
        x_data = parse_input(x_raw)
    except ValueError:
        flash("Invalid characters in X input. Please enter only numeric values.")
        return render_template('index.html', x_input=x_raw, y_input=y_raw)

    try:
        y_data = parse_input(y_raw)
    except ValueError:
        flash("Invalid characters in Y input. Please enter only numeric values.")
        return render_template('index.html', x_input=x_raw, y_input=y_raw)

    if not x_data or not y_data:
        flash("Input fields cannot be empty.")
        return render_template('index.html', x_input=x_raw, y_input=y_raw)

    if len(x_data) != len(y_data):
        flash(f"Mismatched data sizes: X has {len(x_data)} items, but Y has {len(y_data)} items.")
        return render_template('index.html', x_input=x_raw, y_input=y_raw)

    if len(x_data) < 2:
        flash("Simple linear regression requires at least 2 data points.")
        return render_template('index.html', x_input=x_raw, y_input=y_raw)

    # Convert to numpy arrays for calculation
    X = np.array(x_data, dtype=float)
    Y = np.array(y_data, dtype=float)
    n = len(X)

    # Regression Math
    x_mean = np.mean(X)
    y_mean = np.mean(Y)

    # Slope (m) and Intercept (c)
    numerator = np.sum((X - x_mean) * (Y - y_mean))
    denominator = np.sum((X - x_mean) ** 2)

    # Handle case where all X values are identical (division by zero)
    if denominator == 0:
        flash("Cannot perform regression if all X values are identical (vertical line).")
        return render_template('index.html', x_input=x_raw, y_input=y_raw)

    m = numerator / denominator
    c = y_mean - m * x_mean

    # R-squared
    y_pred = m * X + c
    ss_tot = np.sum((Y - y_mean) ** 2)
    ss_res = np.sum((Y - y_pred) ** 2)

    if ss_tot == 0:
        r_squared = 1.0 if np.allclose(Y, y_pred) else 0.0
    else:
        r_squared = 1.0 - (ss_res / ss_tot)

    r_squared = max(0.0, min(1.0, r_squared))

    # Format equation string
    sign = "+" if c >= 0 else "-"
    eq_str = f"Y = {m:.4f}X {sign} {abs(c):.4f}"

    # Generate Matplotlib visualization (Light Theme / Pastel Palette)
    plt.figure(figsize=(8, 5.5), facecolor='#fbfbfc')
    ax = plt.axes()
    ax.set_facecolor('#ffffff')

    # Pastel Color Customization
    scatter_color = '#70d6ff'      # Pastel sky blue
    line_color = '#ff70a6'         # Pastel pink/coral
    grid_color = '#f1f3f5'
    spine_color = '#dee2e6'
    text_color = '#2b3a4a'         # Dark slate grey for readable text

    # Plot original scatter data
    plt.scatter(X, Y, color=scatter_color, edgecolors='#4e89ae', linewidths=0.8, s=90, label='Data Points', zorder=5)

    # Plot regression line
    x_min, x_max = np.min(X), np.max(X)
    x_pad = (x_max - x_min) * 0.08 if x_max != x_min else 1.0
    x_line = np.linspace(x_min - x_pad, x_max + x_pad, 100)
    y_line = m * x_line + c
    plt.plot(x_line, y_line, color=line_color, linewidth=3.2, label='Regression Line', zorder=4)

    # Style labels and title
    plt.title('Simple Linear Regression Fit', fontsize=14, color=text_color, fontweight='bold', pad=15)
    plt.xlabel('Independent Variable (X)', fontsize=11, color=text_color, labelpad=10)
    plt.ylabel('Dependent Variable (Y)', fontsize=11, color=text_color, labelpad=10)

    # Style spines and grid
    for spine in ax.spines.values():
        spine.set_color(spine_color)
        spine.set_linewidth(1.2)
    ax.tick_params(colors=text_color, labelsize=9)
    plt.grid(True, linestyle='--', color=grid_color, alpha=0.9, zorder=1)

    # Add legend
    legend = plt.legend(facecolor='#ffffff', edgecolor=spine_color, labelcolor=text_color, loc='best')
    legend.get_frame().set_linewidth(1.0)

    # Adjust layout and save
    plt.tight_layout()
    plot_filename = 'regression_plot.png'
    plot_filepath = os.path.join(PLOTS_DIR, plot_filename)
    
    if os.path.exists(plot_filepath):
        try:
            os.remove(plot_filepath)
        except OSError:
            pass

    plt.savefig(plot_filepath, dpi=120, facecolor='#fbfbfc', edgecolor='none')
    plt.close()

    # Determine correlation rating
    if r_squared >= 0.9:
        rating = "Excellent Fit (Very Strong Correlation)"
        rating_color = "#38b000" # Soft dark green
    elif r_squared >= 0.7:
        rating = "Good Fit (Strong Correlation)"
        rating_color = "#0096c7" # Soft dark cyan/blue
    elif r_squared >= 0.4:
        rating = "Moderate Fit (Moderate Correlation)"
        rating_color = "#f77f00" # Soft orange
    elif r_squared > 0.05:
        rating = "Weak Fit (Low Correlation)"
        rating_color = "#e63946" # Soft red
    else:
        rating = "No Correlation (Random Scatter)"
        rating_color = "#d62828" # Deep soft red

    return render_template(
        'result.html',
        slope=f"{m:.4f}",
        intercept=f"{c:.4f}",
        r_squared=f"{r_squared:.4f}",
        r_squared_pct=f"{r_squared * 100:.2f}%",
        equation=eq_str,
        n=n,
        rating=rating,
        rating_color=rating_color,
        timestamp=int(time.time()),
        x_raw=x_raw,
        y_raw=y_raw
    )

if __name__ == '__main__':
    # Run on port 5051 to avoid conflicts with version 1
    app.run(debug=True, use_reloader=False, port=5051)
