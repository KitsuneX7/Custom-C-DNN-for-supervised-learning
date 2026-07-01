import numpy as np
from scipy.special import erf

# Domain and max error
# for sigmoid and tanh
domains = [[-3, -1.8], [-1.8, -1], [-1, -0.4], [-0.4, 0], [0, 0.4], [0.4, 1], [1, 1.8], [1.8, 3]]
epsilon = 0.005  # Maximum allowable error bound

# Global vars
total_intervals = 0
all_x_spline = []
all_slopes = []
all_intercepts = []

# Non-linear activation functions
def myGeLU(x):
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))

def mySigmoid(x):
    return 1 / (1 + np.exp(-x))

def myTanh(x):
    return np.tanh(x)

def myGeLUSecondDer(x):
    numerator = 2.0 - (x ** 2)
    denominator = np.sqrt(2.0 * np.pi)
    gaussian_density = np.exp(-(x ** 2) / 2.0)
    return (numerator / denominator) * gaussian_density

def mySigmoidSecondDer(x):
    t = 1 / (1 + np.exp(-x))
    return t * (1 - t) * (1 - 2 * t)

def myTanhSecondDer(x):
    t = np.tanh(x)
    return -2 * t * (1 - t**2)

ans = int(input("Choose function to approximate (0 - GeLU / 1 - sigmoid / 2 - tanh): "))
if ans != 0 and ans != 1 and ans != 2:
    print("Invalid choice!")
    while ans != 0 and ans != 1 and ans != 2:
        ans = int(input("Choose function to approximate (0 - GeLU / 1 - sigmoid / 2 - tanh): "))
        print("Invalid choice!")

for domain in domains:
    x_min, x_max = domain[0], domain[1]
    # Maximum value of the second derivative in the linear space
    x_space = np.linspace(x_min, x_max, 1000)
    if ans == 0:
        M2 = np.max(np.abs(myGeLUSecondDer(x_space)))
    elif ans == 1:
        M2 = np.max(np.abs(mySigmoidSecondDer(x_space)))
    else:
        M2 = np.max(np.abs(myTanhSecondDer(x_space)))

    h = np.sqrt((8 * epsilon) / M2)

    # Generating the points
    num_intervals = int(np.ceil((x_max - x_min) / h))
    num_points = num_intervals + 1
    x_spline = np.linspace(x_min, x_max, num_points)
    if ans == 0:
        y_spline = myGeLU(x_spline)
    elif ans == 1:
        y_spline = mySigmoid(x_spline)
    else:
        y_spline = myTanh(x_spline)

    # Slopes and intercepts
    slopes = np.diff(y_spline) / np.diff(x_spline)
    intercepts = y_spline[:-1]  # The 'c' value at the start of each interval

    # Add it all up for the overview later on
    total_intervals += num_intervals
    all_x_spline.extend(x_spline)
    all_slopes.extend(slopes)
    all_intercepts.extend(intercepts)

    # Print for each subsection
    print(f"// Total Intervals: {num_intervals}")
    print(f"const float x_nodes[{num_points}] = " + "{" + ", ".join(map(str, x_spline)) + "};")
    print(f"const float m_coefficients[{num_intervals}] = " + "{" + ", ".join(map(str, slopes)) + "};")
    print(f"const float c_coefficients[{num_intervals}] = " + "{" + ", ".join(map(str, intercepts)) + "};")

# Clean up intervals
all_x_spline = sorted(list(set(all_x_spline)))

# Final print
print(f"// Total Intervals: {total_intervals}")
print(f"const float x_nodes[{len(all_x_spline)}] = " + "{" + ", ".join(map(str, all_x_spline)) + "};")
print(f"const float m_coefficients[{total_intervals}] = " + "{" + ", ".join(map(str, all_slopes)) + "};")
print(f"const float c_coefficients[{total_intervals}] = " + "{" + ", ".join(map(str, all_intercepts)) + "};")