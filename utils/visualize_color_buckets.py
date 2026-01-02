"""
Visualize color buckets on a 360-degree color wheel.

Shows the hue ranges for each color bucket with boundaries marked.
Also shows achromatic (black/gray/white) boundaries on a saturation-lightness plot.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import hsv_to_rgb
import colorsys

# Color bucket boundaries (in hue, normalized 0-1)
RED_YELLOW_BOUNDARY = 0.0944  # 34°
YELLOW_GREEN_BOUNDARY = 0.2222  # 80°
GREEN_BLUE_BOUNDARY = 0.4722  # 170°
BLUE_OTHER_BOUNDARY = 0.8056  # 290°
OTHER_RED_BOUNDARY = 0.8333  # 300°

# Achromatic color boundaries (saturation and lightness thresholds)
S_GRAY_MAX = 0.15  # Maximum saturation for achromatic (gray/black/white)
L_WHITE_MIN = 0.95  # Minimum lightness for white
L_BLACK_MAX = 0.2  # Maximum lightness for black

# Bucket names and colors for legend
BUCKET_COLORS = {
    'Red': '#FF0000',
    'Yellow': '#FFFF00',
    'Green': '#00FF00',
    'Blue': '#0000FF',
    'Other': '#FF00FF',  # Magenta/purple
}

def create_achromatic_plot():
    """
    Create a saturation-lightness plot showing black, gray, and white boundaries.
    """
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
    ax.set_xlabel('Saturation', fontsize=10, fontweight='bold')
    ax.set_ylabel('Lightness', fontsize=10, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Create gradient for achromatic region (S <= S_GRAY_MAX, L from 0 to 1)
    # Create a meshgrid for the achromatic region
    s_resolution = 200
    l_resolution = 200
    s_achromatic = np.linspace(0, S_GRAY_MAX, s_resolution)
    l_values = np.linspace(0, 1, l_resolution)
    
    # Create 2D arrays for the gradient
    S_mesh, L_mesh = np.meshgrid(s_achromatic, l_values)
    
    # Create grayscale colors: convert lightness to RGB grayscale
    # For grayscale, R=G=B=lightness value
    grayscale_colors = np.zeros((l_resolution, s_resolution, 3))
    for i in range(l_resolution):
        gray_value = l_values[i]
        grayscale_colors[i, :, 0] = gray_value  # R
        grayscale_colors[i, :, 1] = gray_value  # G
        grayscale_colors[i, :, 2] = gray_value  # B
    
    # Display the gradient using imshow
    ax.imshow(grayscale_colors, extent=[0, S_GRAY_MAX, 0, 1], 
              aspect='auto', origin='lower', interpolation='bilinear')
    
    # Draw grid after gradient so it's visible
    ax.grid(True, alpha=0.3)
    
    # Draw boundary lines
    # Saturation boundary for achromatic
    ax.axvline(x=S_GRAY_MAX, color='black', linewidth=2, linestyle='--', 
               label=f'Saturation threshold: {S_GRAY_MAX}')
    
    # Lightness boundaries
    ax.axhline(y=L_WHITE_MIN, color='black', linewidth=2, linestyle='--',
               label=f'White threshold: L >= {L_WHITE_MIN}')
    ax.axhline(y=L_BLACK_MAX, color='black', linewidth=2, linestyle='--',
               label=f'Black threshold: L <= {L_BLACK_MAX}')
    
    # Chromatic region: S > S_GRAY_MAX (all lightness values)
    ax.fill_between([S_GRAY_MAX, 1], [0, 0], [1, 1],
                    color='lightblue', alpha=0.1, edgecolor='blue', linewidth=1)
    
    # Add title
    ax.set_title('Achromatic Color Boundaries\n(Saturation vs Lightness)', 
                 fontsize=12, fontweight='bold', pad=15)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    
    plt.tight_layout()
    return fig, ax


def create_color_wheel(size=800, center_radius=0.3):
    """
    Create a color wheel visualization with bucket boundaries.
    
    Args:
        size: Size of the image in pixels
        center_radius: Radius of the center circle (0-1, relative to image)
    """
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Create polar grid
    n_angles = 360
    n_radii = 100
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    radii = np.linspace(center_radius, 1.0, n_radii)
    
    # Create meshgrid
    theta, r = np.meshgrid(angles, radii)
    
    # Convert hue from 0-360° to 0-1 range (HLS)
    # Note: colorsys uses 0-1 for hue, where 0=red, 1/6=yellow, 1/3=green, etc.
    # But we need to map our boundaries correctly
    hue_normalized = theta / (2 * np.pi)
    
    # Create RGB colors from HSV (saturation=1, value=1 for full color)
    # HSV hue is already in 0-1 range
    hsv_colors = np.zeros((n_radii, n_angles, 3))
    hsv_colors[:, :, 0] = hue_normalized  # Hue
    hsv_colors[:, :, 1] = 1.0  # Saturation
    hsv_colors[:, :, 2] = 1.0  # Value
    
    # Convert HSV to RGB
    rgb_colors = hsv_to_rgb(hsv_colors)
    
    # Create the color wheel using pcolormesh in polar coordinates
    # Convert to Cartesian
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # Plot the color wheel
    ax.pcolormesh(x, y, rgb_colors, shading='gouraud')
    
    # Draw bucket boundaries
    boundary_angles_deg = [
        0,  # Red start (0°)
        34,  # Red-Yellow boundary
        80,  # Yellow-Green boundary
        170,  # Green-Blue boundary
        290,  # Blue-Other boundary
        300,  # Other-Red boundary
        360,  # Red end (360°)
    ]
    
    boundary_angles_rad = [np.deg2rad(angle) for angle in boundary_angles_deg]
    
    # Draw boundary lines
    for angle_rad in boundary_angles_rad[1:-1]:  # Skip first and last (0° and 360° are same)
        x_line = np.array([center_radius, 1.0]) * np.cos(angle_rad)
        y_line = np.array([center_radius, 1.0]) * np.sin(angle_rad)
        ax.plot(x_line, y_line, 'k-', linewidth=3, alpha=0.8)
    
    # Draw center circle (white)
    center_circle = mpatches.Circle((0, 0), center_radius, facecolor='white', 
                                    edgecolor='black', linewidth=2)
    ax.add_patch(center_circle)
    
    # Add title
    ax.text(0, -1.2, 'Color Bucket Boundaries', ha='center', va='top',
            fontsize=12, fontweight='bold')
    
    # Add legend (smaller and more compact)
    legend_elements = [
        mpatches.Patch(facecolor='#FF0000', label='Red: 0-34°, 300-360°'),
        mpatches.Patch(facecolor='#FFFF00', label='Yellow: 34-80°'),
        mpatches.Patch(facecolor='#00FF00', label='Green: 80-170°'),
        mpatches.Patch(facecolor='#0000FF', label='Blue: 170-290°'),
        mpatches.Patch(facecolor='#FF00FF', label='Other: 290-300°'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8,
              framealpha=0.9)
    
    plt.tight_layout()
    return fig, ax


if __name__ == '__main__':
    # Create both visualizations
    fig1, ax1 = create_color_wheel()
    
    # Save the color wheel
    output_path = 'color_bucket_wheel.png'
    plt.figure(fig1.number)
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Color wheel saved to {output_path}")
    
    output_path_pdf = 'color_bucket_wheel.pdf'
    plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
    print(f"Color wheel saved to {output_path_pdf}")
    
    # Create achromatic plot
    fig2, ax2 = create_achromatic_plot()
    
    # Save the achromatic plot
    output_path_achromatic = 'color_bucket_achromatic.png'
    plt.figure(fig2.number)
    plt.savefig(output_path_achromatic, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Achromatic boundaries plot saved to {output_path_achromatic}")
    
    output_path_achromatic_pdf = 'color_bucket_achromatic.pdf'
    plt.savefig(output_path_achromatic_pdf, bbox_inches='tight', facecolor='white')
    print(f"Achromatic boundaries plot saved to {output_path_achromatic_pdf}")
    
    plt.show()

