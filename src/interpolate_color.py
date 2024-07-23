import matplotlib.pyplot as plt
import numpy as np

def interpolate_color(color0, color1, x):
    """
    Interpolates between two RGB colors.

    Args:
        color0 (tuple): The RGB color at x = 0. Should be a tuple of three integers (R, G, B).
        color1 (tuple): The RGB color at x = 1. Should be a tuple of three integers (R, G, B).
        x (float): The interpolation factor, where 0 <= x <= 1.

    Returns:
        tuple: The interpolated RGB color.
    """
    if not (0 <= x <= 1):
        raise ValueError("The interpolation factor x must be between 0 and 1.")

    r = int(color0[0] + (color1[0] - color0[0]) * x)
    g = int(color0[1] + (color1[1] - color0[1]) * x)
    b = int(color0[2] + (color1[2] - color0[2]) * x)

    return (r, g, b)

# Example usage:
#color_at_0 = (12, 142, 210)  # Left
#color_at_1 = (186, 0, 32)  # Right
#x_value = 0.5  # Midpoint
#interpolated_color = interpolate_color(color_at_0, color_at_1, x_value)

def plot_color_gradient(color0, color1, num_steps=1000):
    """
    Plots a color gradient from color0 to color1.

    Args:
        color0 (tuple): The RGB color at x = 0. Should be a tuple of three integers (R, G, B).
        color1 (tuple): The RGB color at x = 1. Should be a tuple of three integers (R, G, B).
        num_steps (int): The number of steps in the gradient.
    """
    gradient = np.zeros((num_steps, 3), dtype=int)
    
    for i in range(num_steps):
        x = i / (num_steps - 1)
        gradient[i] = interpolate_color(color0, color1, x)

    # Create a gradient image
    gradient_image = np.tile(gradient, (50, 1, 1))

    # Plot the gradient image
    plt.figure(figsize=(10, 2))
    plt.imshow(gradient_image, aspect='auto')
    plt.axis('off')
    plt.show()

# Example usage:
#color_at_0 = (12, 142, 210)  # Left
#color_at_1 = (186, 0, 32)  # Right
#plot_color_gradient(color_at_0, color_at_1)