import numpy as np
import bezier
import matplotlib.pyplot as plt
from glaze import render
import math

OUTPUT_FORMATS = ["png"]

def render_curve(curve, curve_name = "test"):
    plt.figure()
    render(np.array([curve]))
    for output_format in OUTPUT_FORMATS:
        plt.savefig("curve.{}".format(output_format))
        #file_name = "curve-{}.{}".format(curve_name, output_format)
        #plt.savefig(os.path.join("outputs", file_name))  
    plt.close()  

def get_contour_length(contour):
    length = 0
    transposed_contour = np.transpose(contour, axes=(0, 2, 1)).astype(np.float64)
    for curve in transposed_contour:
        curve = bezier.Curve(curve, degree=2)
        length += curve.length
    return length


def get_delta_thetas(contour):
    # return a list of angle changes between each bezier curve.
    angles = []
    delta_angles = []
    for curve in contour:
        #render_curve(curve)
        delta_x = curve[2][0] - curve[0][0]
        delta_y = curve[2][1] - curve[0][1]
        theta = np.arctan2(delta_y, delta_x)
        angles.append(theta)
    delta_x = contour[-1][2][0] - contour[0][0][0]
    delta_y = contour[-1][2][1] - contour[0][0][1]
    theta = np.arctan2(delta_y, delta_x)
    angles.append(theta)

    delta_angles = [abs(angles[i] - angles[i - 1]) for i in range(0, len(angles))]
    filtered_delta_angles = correct_delta_thetas_above_threshold(delta_angles)
    plot_delta_thetas(angles, filtered_delta_angles)

    return filtered_delta_angles

def integral_of_delta_thetas(delta_angles):
    #Trapzoidal rule
    curve_num = [0]
    integral_vals = [0]
    integral = 0
    for i in range(len(delta_angles) - 1):
        integral += (delta_angles[i] + delta_angles[i+1])/2
        curve_num.append(i)
        integral_vals.append(integral)
    return integral_vals

def plot_integral(delta_angles):
    integral = 0
    x = [0]
    y = [0]
    for i in range(1, len(delta_angles)):
        integral += (delta_angles[i-1] + delta_angles[i])/2
        x.append(i)
        y.append(integral)
    plt.figure()
    plt.plot(x,y)
    plt.title("Integral values")
    plt.xlabel("Curve number")
    plt.ylabel("Integral_value")
    plt.savefig("delta_angles_integral.png")
    plt.show()
    


def get_fractional_locations(integral_vals, integral_step_length):
    # for i in range(len(delta_thetas) - 1):
    #     if integral += (delta_angles[i] + delta_angles[i+1])/2
    loc = 0
    locs = []
    loc_integral = 0
    for i in range(len(integral_vals) - 1):
        while i <= loc <= i+1:
            delta_loc = (loc_integral - integral_vals[i])/(integral_vals[i+1] - integral_vals[i])
            loc = i + delta_loc
            if i <= loc <= i+1:
                locs.append(loc)
                loc_integral += integral_step_length
    locs[-1] = len(integral_vals) - 1
    return locs


def generate_contour_distribution(contour, num_points = 20):
    delta_thetas = get_delta_thetas(contour)
    delta_thetas_integral_vals = integral_of_delta_thetas(delta_thetas)
    integral_step_length = delta_thetas_integral_vals[-1]/num_points
    plot_integral(delta_thetas)
    fractional_locations_of_integral_step_values = get_fractional_locations(delta_thetas_integral_vals, integral_step_length)
    return fractional_locations_of_integral_step_values
    
    #normalized_delta_thetas = delta_thetas/max(delta_thetas)
    #print("Normalized angle changes")
    #print(normalized_delta_thetas)

def correct_delta_thetas_above_threshold(delta_angles, threshold = 1.5*np.pi):
    for i in range(len(delta_angles)):
        if delta_angles[i] > threshold:
            delta_angles[i] = abs(delta_angles[i] - 2*np.pi)
        if delta_angles[i] < -1*threshold:
            delta_angles[i] = abs(delta_angles[i] + 2*np.pi)
    return delta_angles

def plot_delta_thetas(angles, delta_angles):
    plt.figure()
    plt.plot(range(len(angles)), [angle*(180/np.pi) for angle in angles])
    plt.title("Thetas")
    plt.xlabel("Curve number")
    plt.ylabel("Angle between points (deg)")
    plt.savefig("thetas.png")
    plt.figure()
    plt.plot(range(len(delta_angles)), [delta_angle*(180/np.pi) for delta_angle in delta_angles])
    plt.title("Delta thetas")
    plt.ylabel("Change in angle between points (deg)")
    plt.xlabel("Curve number")
    plt.savefig("deltathetas.png")
    plt.figure()
    plt.plot(range(len(angles)), angles)
    plt.title("Thetas")
    plt.xlabel("Curve number")
    plt.ylabel("Angle between points (rad)")
    plt.savefig("thetas_rad.png")
    plt.figure()
    plt.plot(range(len(delta_angles)), delta_angles)
    plt.title("Delta thetas")
    plt.ylabel("Change in angle between points (rad)")
    plt.xlabel("Curve number")
    plt.savefig("deltathetas_rad.png")
    plt.show()