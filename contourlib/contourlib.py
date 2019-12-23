import numpy as np
import bezier
import matplotlib.pyplot as plt
from .render import render
import math

OUTPUT_FORMATS = ["png"]
THRESHOLD = np.pi
EPSILON = 0.01
CORNER_THRESHOLD = 0.1

def get_contour_corners(contour):
    corners = []
    start_points = []
    end_points = []
    epsilon_start_points = []
    epsilon_end_points = []
    for curve in contour:
        bezier_curve = bezier.Curve(curve, degree=2)
        start_point_x = curve[0][0]
        start_point_y = curve[1][0]
        start_points.append([start_point_x, start_point_y])
        end_point_x = curve[0][2]
        end_point_y = curve[1][2]
        end_points.append([end_point_x, end_point_y])
        epsilon_start_point = bezier_curve.evaluate(EPSILON).flatten()
        epsilon_end_point = bezier_curve.evaluate(1-EPSILON).flatten()
        epsilon_start_points.append(epsilon_start_point)
        epsilon_end_points.append(epsilon_end_point)
    
    for i in range(len(start_points) - 1):
        delta_x_end = end_points[i][0] - epsilon_end_points[i][0]
        delta_y_end = end_points[i][1] - epsilon_end_points[i][1]
        theta_1 = np.arctan2(delta_y_end, delta_x_end)
        delta_x_start = epsilon_start_points[i+1][0] - start_points[i+1][0]
        delta_y_start = epsilon_start_points[i+1][1] - start_points[i+1][1]
        theta_2 = np.arctan2(delta_y_start, delta_x_start)
        delta_theta = abs(theta_2 - theta_1)
        if delta_theta > THRESHOLD:
            delta_theta = abs(delta_theta - 2*np.pi)
        if delta_theta < -1*THRESHOLD:
            delta_theta = abs(delta_theta + 2*np.pi)
        if delta_theta > CORNER_THRESHOLD:
            corners.append(start_points[i+1])

    delta_x_end = end_points[-1][0] - epsilon_end_points[-1][0]
    delta_y_end = end_points[-1][1] - epsilon_end_points[-1][1]
    theta_1 = np.arctan2(delta_y_end, delta_x_end)
    delta_x_start = epsilon_start_points[0][0] - start_points[0][0]
    delta_y_start = epsilon_start_points[0][1] - start_points[0][1]
    theta_2 = np.arctan2(delta_y_start, delta_x_start)
    delta_theta = abs(theta_2 - theta_1)
    if delta_theta > THRESHOLD:
        delta_theta = abs(delta_theta - 2*np.pi)
    if delta_theta < -1*THRESHOLD:
        delta_theta = abs(delta_theta + 2*np.pi)
    if delta_theta > CORNER_THRESHOLD:
        corners.append(start_points[0])
    plt.figure()
    plt.scatter(*zip(*corners))
    plt.savefig("test.png")
    return corners

def get_closest_point(new_contour, corner):
    j = 0
    start_point_x = new_contour[0][0][0]
    start_point_y = new_contour[0][0][1]
    shortest_distance = np.sqrt( (corner[0]-start_point_x)**2 + (corner[1]-start_point_y)**2 )
    for i in range(1, len(new_contour)):
        start_point_x = new_contour[i][0][0]
        start_point_y = new_contour[i][0][1]
        distance = np.sqrt( (corner[0]-start_point_x)**2 + (corner[1]-start_point_y)**2 )
        if distance < shortest_distance:
            shortest_distance = distance
            j = i
    return j

def get_snapped_corner_contour(contour, new_contour):
    contour_corners = get_contour_corners(contour)
    for corner in contour_corners:
        startpoint_index_to_swap = get_closest_point(new_contour, corner)
        new_contour[startpoint_index_to_swap][0][0] = corner[0]
        new_contour[startpoint_index_to_swap][0][1] = corner[1]
        new_contour[startpoint_index_to_swap-1][2][0] = corner[0]
        new_contour[startpoint_index_to_swap-1][2][1] = corner[1]
    for i in range(len(new_contour)):
        new_contour[i][1] = [(new_contour[i][0][0] + new_contour[i][2][0])/2, (new_contour[i][0][1] + new_contour[i][2][1])/2]
    return np.array(new_contour)

def make_fixed_num_contour(start_points, end_points):
        fixed_num_contour = []
        for start_point, end_point in zip(start_points, end_points):
            off_point = [(start_point[0] + end_point[0])/2, (start_point[1] + end_point[1])/2]
            #off_point = create_gaussian_noise(off_point)
            curve = [start_point, off_point, end_point]
            fixed_num_contour.append(curve)
        return fixed_num_contour

def create_gaussian_noise(off_point, scale=0.05):
    noise = np.random.normal(scale=scale, size=2)
    noisy_point = [off_point[0] + noise[0], off_point[1] + noise[1]]
    return noisy_point

def swap_bad_contours(contours):
    filtered_contours = []
    for contour in contours:
        transposed_contour = np.transpose(contour, axes=(0, 2, 1)).astype(np.float64)
        for i in range(len(contour)):
            end_point_x = transposed_contour[i][0][2]
            end_point_y = transposed_contour[i][1][2]
            if i != len(transposed_contour) - 1:
                next_start_point_x = transposed_contour[i+1][0][0]
                next_start_point_y = transposed_contour[i+1][1][0]
            else:
                next_start_point_x = transposed_contour[0][0][0]
                next_start_point_y = transposed_contour[0][1][0]
            if not (end_point_x == next_start_point_x and end_point_y==next_start_point_y):
                if i != len(transposed_contour) - 1:
                    transposed_contour[i+1] = np.flip(transposed_contour[i+1], axis=1)
                else:
                    transposed_contour[0] = np.flip(transposed_contour[0], axis=1)
        transposed_contour = np.transpose(transposed_contour, axes=(0, 2, 1)).astype(np.float64)
        filtered_contours.append(transposed_contour)
    return filtered_contours


def render_curve(curve, curve_name = "test"):
    plt.figure()
    render(np.array([curve]))
    for output_format in OUTPUT_FORMATS:
        plt.savefig("curve.{}".format(output_format))
        #file_name = "curve-{}.{}".format(curve_name, output_format)
        #plt.savefig(os.path.join("outputs", file_name))
    plt.close()

def render_contour(contour, contour_name = "test"):
    plt.figure()
    render(np.array(contour))
    for output_format in OUTPUT_FORMATS:
        plt.savefig("contour.{}".format(output_format))
        #file_name = "contour-{}.{}".format(contour_name, output_format)
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
        delta_x = curve[2][0] - curve[0][0]
        delta_y = curve[2][1] - curve[0][1]
        theta = np.arctan2(delta_y, delta_x)
        angles.append(theta)
    delta_x = contour[-1][2][0] - contour[0][0][0]
    delta_y = contour[-1][2][1] - contour[0][0][1]
    theta = np.arctan2(delta_y, delta_x)
    angles.append(theta)

    delta_angles = [abs(angles[i-1] - angles[i]) for i in range(0, len(angles))]
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
    # plt.show()



def get_fractional_locations(integral_vals, integral_step_length):
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
            else:
                loc = (i+1)
                break
    locs[-1] = len(integral_vals) - 1
    return locs


def generate_contour_distribution(contour, num_points = 20):
    delta_thetas = get_delta_thetas(contour)
    delta_thetas_integral_vals = integral_of_delta_thetas(delta_thetas)
    integral_step_length = delta_thetas_integral_vals[-1]/num_points
    plot_integral(delta_thetas)
    fractional_locations_of_integral_step_values = get_fractional_locations(delta_thetas_integral_vals, integral_step_length)
    return fractional_locations_of_integral_step_values

def correct_delta_thetas_above_threshold(delta_angles):
    for i in range(len(delta_angles)):
        if delta_angles[i] > THRESHOLD:
            delta_angles[i] = abs(delta_angles[i] - 2*np.pi)
        if delta_angles[i] < -1*THRESHOLD:
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
    # plt.show()
