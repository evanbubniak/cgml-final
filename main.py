# encoding: utf-8

import os
import bezier
import numpy as np
from glaze import read_json, render
import matplotlib.pyplot as plt
from math import isclose
import argparse

UPPERCASES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
LOWERCASES = [character.lower() for character in UPPERCASES]
NUMERALS = ["zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine"]

SPECIALS = ["exclam",
    "numbersign",
    "dollar",
    "percent",
    "ampersand",
    "asterisk",
    "question",
    "at"]
CHARACTER_SET = UPPERCASES + LOWERCASES + NUMERALS + SPECIALS

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output_format", nargs="*", default = ["png"],
    help="Output format of all renderings. Any output format supported by matplotlib.pyplot.savefig works here. Tip: PNG for pixel image, EPS for vector.")
parser.add_argument("-g", "--glyph", nargs="*", default = ["o"],
    help = "list of glyphs to render")
parser.add_argument("-p", "--points", nargs="?", type=int, default = 20,
    help = "Num points to render")
parser.add_argument("-f", "--font", type=str, nargs="?", default="arial",
    help = "Font to use")
args = parser.parse_args()
if args.glyph == ["all"]:
    args.glyph = CHARACTER_SET

OUTPUT_FORMATS = args.output_format

def render_curve(curve, curve_name = "test"):
    plt.figure()
    render(np.array([curve]))
    for output_format in OUTPUT_FORMATS:
        file_name = "curve-{}.{}".format(curve_name, output_format)
        plt.savefig(os.path.join("outputs", file_name))
    plt.close()

def get_contour_length(contour):
    length = 0
    transposed_contour = np.transpose(contour, axes=(0, 2, 1)).astype(np.float64)
    for curve in transposed_contour:
        curve = bezier.Curve(curve, degree=2)
        length += curve.length
    return length

def make_output_dirs():
    output_types = ["raw", "raw-straight-line", "fixed-straight-line"]
    for output_type in output_types:
        output_path = os.path.join("outputs", output_type)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

def create_gaussian_noise(off_point, scale=0.05):
    noise = np.random.normal(scale=scale, size=2)
    noisy_point = [off_point[0] + noise[0], off_point[1] + noise[1]]
    return noisy_point

def get_angle_changes(contour):
    # return a list of angle changes between each bezier curve.
    angles = []
    delta_angles = []
    for curve in contour:
        delta_x = curve[2][0] - curve[0][0]
        delta_y = curve[2][1] - curve[0][1]
        theta = np.arctan2(delta_y, delta_x)
        angles.append(theta)

    # delta_angles = [angles[i] - angles[i - 1] for i in range(0, len(angles))]

    for i in range(0, len(angles)):
        delta_angles.append(angles[i] - angles[i - 1])
        if i == 0:
            pass
        else:
            if delta_angles[i] > 5:
                delta_angles[i] = delta_angles[i] - 6.2
            if delta_angles[i] < -5:
                delta_angles[i] = delta_angles[i] + 6.2

    print("ang", angles)
    print("change", delta_angles)


    plt.figure()
    plt.plot(range(len(angles)), angles)
    plt.title("Thetas")
    plt.xlabel("Curve number")
    plt.ylabel("Angle between points (rad)")
    plt.savefig("thetas.png")
    plt.figure()
    plt.plot(range(len(delta_angles)), delta_angles)
    plt.title("Delta thetas")
    plt.ylabel("Change in angle between points (rad)")
    plt.xlabel("Curve number")
    plt.savefig("deltathetas.png")
    plt.show()
    return delta_angles


class Glyph:
    def __init__(self, font_name, char_name):
        self.glyph = None
        self.char_name = char_name
        if char_name.isupper():
            self.char_name += "-uppercase"
        font_path = os.path.join("fonts", "json", "{}.json".format(font_name.lower()))
        font = read_json(font_path)
        for glyph in font:
            if glyph[1] == char_name:
                self.glyph = glyph
        if self.glyph is None:
            raise Exception("Character not found in font")
        self.font_name = self.glyph[0]
        self.contours = self.glyph[2]
        self.num_contours = len(self.contours)


    def render_raw_glyph(self):
        # Just renders directly using Glaze and saves it.
        plt.figure()
        render(self.contours)
        for output_format in OUTPUT_FORMATS:
            file_name = "{}-{}-raw-glyph.{}".format(self.font_name, self.char_name, output_format)
            plt.savefig(os.path.join("outputs", "raw", file_name))
        plt.close()

    def render_straight_line_glyph(self):
        '''
        glyph assumed format: [#contours, #beziers, 3, 2, 1]
        - get glyph and raster
        (raster is to compare against)
        - for each contour in the glyph:
        - sample the curves and divide into set # of segments
        - put off-curve points onto the line
        1) generalization of beziers
        2) fix off-curve points
        '''
        contours = self.contours
        for contour in contours:
            for curve in contour:
                for axis in range(len(curve[0])):
                    curve[1][axis] = (curve[0][axis] + curve[2][axis])/2
        plt.figure()
        render(contours)
        for output_format in OUTPUT_FORMATS:
            file_name = "{}-{}-raw-straight-line-glyph.{}".format(self.font_name, self.char_name, output_format)
            plt.savefig(os.path.join("outputs", "raw-straight-line", file_name))
        plt.close()

    def generate_contour_distribution(self, contour):
        angle_changes = get_angle_changes(contour)
        normalized_angle_changes = angle_changes/max(angle_changes)

    def render_fixed_num_bezier(self, num_points = 20):
        for contour in self.contours:
            self.generate_contour_distribution(contour)

    def render_fixed_num_distance_bezier(self, num_points = 20):
        fixed_num_contours = []

        for contour in self.contours:

            fixed_num_contour = []
            start_points = []
            end_points = []

            contour_len = get_contour_length(contour)
            dist_per_point = contour_len/(num_points)
            loc = 0
            len_sum = 0
            transposed_contour = np.transpose(contour, axes=(0, 2, 1)).astype(np.float64)

            first = True
            last = False

            for curve in transposed_contour:
                bezier_curve = bezier.Curve(curve, degree = 2)
                len_sum += bezier_curve.length
                while loc < len_sum or isclose(loc, len_sum):
                    if isclose(loc, contour_len):
                        last = True
                    proportion = 1 - (len_sum - loc)/bezier_curve.length
                    point = bezier_curve.evaluate(proportion).flatten()
                    if first:
                        start_points.append(point)
                    elif last:
                        end_points.append(point)
                    else:
                        start_points.append(point)
                        end_points.append(point)
                    loc += dist_per_point
                    first = False

            for start_point, end_point in zip(start_points, end_points):
                off_point = [(start_point[0] + end_point[0])/2, (start_point[1] + end_point[1])/2]
                #off_point = create_gaussian_noise(off_point)
                curve = [start_point, off_point, end_point]
                fixed_num_contour.append(curve)
            fixed_num_contours.append(np.array(fixed_num_contour))

        plt.figure()
        render(fixed_num_contours)
        for output_format in OUTPUT_FORMATS:
            file_name = "{}-{}-fixed-straight-line-glyph.{}".format(self.font_name, self.char_name, output_format)
            plt.savefig(os.path.join("outputs", "fixed-straight-line", file_name))
        plt.close()

if __name__ == "__main__":
    make_output_dirs()
    for glyph in args.glyph:
        glyph = Glyph(font_name = args.font, char_name=glyph)
        # try:
        #     glyph.render_raw_glyph()
        # except:
        #     print(glyph.char_name)
        #glyph.render_straight_line_glyph()
        glyph.render_fixed_num_bezier(num_points = args.points)
