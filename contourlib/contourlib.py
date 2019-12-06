import numpy as np
import bezier
import matplotlib.pyplot as plt

def get_contour_length(contour):
    length = 0
    transposed_contour = np.transpose(contour, axes=(0, 2, 1)).astype(np.float64)
    for curve in transposed_contour:
        curve = bezier.Curve(curve, degree=2)
        length += curve.length
    return length


def get_angle_changes(contour):
    # return a list of angle changes between each bezier curve.
    angles = []
    delta_angles = []
    for curve in contour:
        delta_x = curve[2][0] - curve[0][0]
        delta_y = curve[2][1] - curve[0][1]
        theta = np.arctan2(delta_y, delta_x)
        angles.append(theta)

    delta_angles = [angles[i] - angles[i - 1] for i in range(0, len(angles))]
    filtered_delta_angles = remove_angle_changes_above_threshold(delta_angles)
    plot_angle_changes(angles, filtered_delta_angles)

    return filtered_delta_angles

def remove_angle_changes_above_threshold(delta_angles, threshold = 1.5*np.pi):
    for i in range(len(delta_angles)):
        if delta_angles[i] > threshold:
            delta_angles[i] = delta_angles[i] - 2*np.pi
        if delta_angles[i] < -1*threshold:
            delta_angles[i] = delta_angles[i] + 2*np.pi
    return delta_angles

def plot_angle_changes(angles, delta_angles):
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
    plt.show()