import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt

def plot_piecewise_approximation(all_x_spline, all_slopes, all_intercepts,
                                 func, x_min=None, x_max=None,
                                 n_points=5000):
    all_x_spline = np.asarray(all_x_spline)
    all_slopes = np.asarray(all_slopes)
    all_intercepts = np.asarray(all_intercepts)

    if x_min is None:
        x_min = all_x_spline[0]
    if x_max is None:
        x_max = all_x_spline[-1]

    x = np.linspace(x_min, x_max, n_points)
    y_true = func(x)
    y_piecewise = np.empty_like(x)

    # Evaluate the piecewise linear approximation
    for i in range(len(all_slopes)):
        left = all_x_spline[i]
        right = all_x_spline[i + 1]

        if i == len(all_slopes) - 1:
            mask = (x >= left) & (x <= right)
        else:
            mask = (x >= left) & (x < right)

        y_piecewise[mask] = (
            all_intercepts[i]
            + all_slopes[i] * (x[mask] - left)
        )

    plt.figure(figsize=(10, 6))
    plt.plot(x, y_true, label="Original function", linewidth=2)
    plt.plot(x, y_piecewise, '--', label="Piecewise approximation", linewidth=2)
    plt.plot(all_x_spline, func(all_x_spline), 'ko', markersize=3,
             label="Spline nodes")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Piecewise Linear Approximation")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


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


ans = int(input("Choose function to approximate (0 - GeLU / 1 - sigmoid / 2 - tanh / 3 - exp): "))
if ans != 0 and ans != 1 and ans != 2 and ans != 3:
    print("Invalid choice!")
    while ans != 0 and ans != 1 and ans != 2 and ans != 3:
        ans = int(input("Choose function to approximate (0 - GeLU / 1 - sigmoid / 2 - tanh / 3 - exp): "))
        print("Invalid choice!")


# Domain and max error
if ans != 3: 
    domains = [[-3, -1.8], [-1.8, -1], [-1, -0.4], [-0.4, 0], [0, 0.4], [0.4, 1], [1, 1.8], [1.8, 3]]
else:
    domains = [[-6,-3], [-3, -1], [-1, 0]]

epsilon = 0.005  # Maximum allowable error bound

for domain in domains:
    x_min, x_max = domain[0], domain[1]
    # Maximum value of the second derivative in the linear space
    x_space = np.linspace(x_min, x_max, 1000)
    if ans == 0:
        M2 = np.max(np.abs(myGeLUSecondDer(x_space)))
    elif ans == 1:
        M2 = np.max(np.abs(mySigmoidSecondDer(x_space)))
    elif ans == 2:
        M2 = np.max(np.abs(myTanhSecondDer(x_space)))
    else:
        M2 = np.max(np.abs(np.exp(x_space)))

    h = np.sqrt((8 * epsilon) / M2)

    # Generating the points
    num_intervals = int(np.ceil((x_max - x_min) / h))
    num_points = num_intervals + 1
    x_spline = np.linspace(x_min, x_max, num_points)
    if ans == 0:
        y_spline = myGeLU(x_spline)
    elif ans == 1:
        y_spline = mySigmoid(x_spline)
    elif ans == 2:
        y_spline = myTanh(x_spline)
    else:
        y_spline = np.exp(x_spline)

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

# print spline approximation
if ans == 0:
    plot_piecewise_approximation(all_x_spline, all_slopes, all_intercepts, myGeLU)
elif ans == 1:
    plot_piecewise_approximation(all_x_spline, all_slopes, all_intercepts, mySigmoid)
elif ans == 2:
    plot_piecewise_approximation(all_x_spline, all_slopes, all_intercepts, myTanh)
else:
    plot_piecewise_approximation(all_x_spline, all_slopes, all_intercepts, lambda x : np.exp(x))

if ans == 0:
    # add f(x)=x for x>=3 for GELU
    all_slopes.append(1)
    all_intercepts.append(3)
else:
    all_slopes.append(0)
    all_intercepts.append(1)

hex_symbols = {"0000": "0", "0001": "1", "0010": "2", "0011": "3",
               "0100": "4", "0101": "5", "0110": "6", "0111": "7",
               "1000": "8", "1001": "9", "1010": "A", "1011": "B",
               "1100": "C", "1101": "D", "1110": "E", "1111": "F"}

def float_to_hex(n):
    binary = np.binary_repr(np.float32(n).view(np.int32), width=32)
    hexa = ""
    for i in range(0,len(binary),4):
        hexa += hex_symbols[binary[i:i+4]]
    return "0x"+ hexa

# print lookup table
print("Lookup table: \n")
print("{")
for i in range(len(all_x_spline)):
    if i != len(all_x_spline)-1:
        dot = ","
    else:
        dot = ""
    print("{"+float_to_hex(all_x_spline[i])+","+float_to_hex(all_slopes[i])+","+float_to_hex(all_intercepts[i])+"}"+dot)
print("}\n")