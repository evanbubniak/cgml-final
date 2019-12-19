# encoding: utf-8

import os
import bezier
import numpy as np
import glaze
import matplotlib.pyplot as plt
from math import isclose
import argparse
import contourlib as cl

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
parser.add_argument("-g", "--glyph", nargs="*", default = ["W"],
    help = "list of glyphs to render")
parser.add_argument("-p", "--points", nargs="?", type=int, default = 20,
    help = "Num points to render")
parser.add_argument("-f", "--font", type=str, nargs="?", default="arial",
    help = "Font to use")
args = parser.parse_args()
if args.glyph == ["all"]:
    args.glyph = CHARACTER_SET

OUTPUT_FORMATS = args.output_format

def make_output_dirs():
    output_types = ["raw", "raw-straight-line", "fixed-straight-line", "fixed-num-var-dist-straight-line"]
    for output_type in output_types:
        output_path = os.path.join("outputs", output_type)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

def create_gaussian_noise(off_point, scale=0.05):
    noise = np.random.normal(scale=scale, size=2)
    noisy_point = [off_point[0] + noise[0], off_point[1] + noise[1]]
    return noisy_point

def make_fixed_num_contour(start_points, end_points):
        fixed_num_contour = []
        for start_point, end_point in zip(start_points, end_points):
            off_point = [(start_point[0] + end_point[0])/2, (start_point[1] + end_point[1])/2]
            #off_point = create_gaussian_noise(off_point)
            curve = [start_point, off_point, end_point]
            fixed_num_contour.append(curve)
        return fixed_num_contour

class Glyph:
    def __init__(self, font_name, char_name):
        self.glyph = None
        self.char_name = char_name
        if char_name.isupper():
            self.char_name += "-uppercase"
        font_path = os.path.join("fonts", "json", "{}.json".format(font_name.lower()))
        font = glaze.read_json(font_path)
        for glyph in font:
            if glyph[1] == char_name:
                self.glyph = glyph
        if self.glyph is None:
            raise Exception("Character not found in font")
        self.font_name = self.glyph[0]
        self.contours = cl.swap_bad_contours(self.glyph[2])
        #self.contours = self.glyph[2]
        self.num_contours = len(self.contours)

    def render_raw_glyph(self):
        render_name = "raw-glyph"
        # Just renders directly using Glaze and saves it.
        plt.figure()
        cl.render(self.contours)
        for output_format in OUTPUT_FORMATS:
            file_name = "{}-{}-{}.{}".format(self.font_name, self.char_name, render_name, output_format)
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
        render_name = "raw-straight-line-glyph"
        contours = self.contours
        for contour in contours:
            for curve in contour:
                for axis in range(len(curve[0])):
                    curve[1][axis] = (curve[0][axis] + curve[2][axis])/2
        plt.figure()
        cl.render(contours)
        for output_format in OUTPUT_FORMATS:
            file_name = "{}-{}-{}.{}".format(self.font_name, self.char_name, render_name, output_format)
            plt.savefig(os.path.join("outputs", "raw-straight-line", file_name))
        plt.close()

    def make_fixed_num_var_dist_bezier(self, num_points):
        fixed_num_contours = []
        for contour in self.contours:
            distribution = cl.generate_contour_distribution(contour, num_points)
            start_points = []
            end_points = []
            
            loc = 0
            len_sum = 0
            transposed_contour = np.transpose(contour, axes=(0, 2, 1)).astype(np.float64)

            first = True
            last = False

            j = 0
            for i in range(len(transposed_contour)):
                bezier_curve = bezier.Curve(transposed_contour[i], degree = 2)
                while i <= distribution[j] <= i+1:
                    if distribution[j] != 0 and distribution[j]%1 == 0:
                        proportion = 1.0
                    else:
                        proportion = distribution[j]%1
                    if distribution[j] == len(transposed_contour):
                        last = True
                    point = bezier_curve.evaluate(proportion).flatten()
                    if first:
                        start_points.append(point)
                    elif last:
                        end_points.append(point)
                    else:
                        start_points.append(point)
                        end_points.append(point)                    
                    j+=1
                    if last:
                        break
                    first = False

            fixed_num_contour = make_fixed_num_contour(start_points, end_points)
            fixed_num_contours.append(np.array(fixed_num_contour))
        return fixed_num_contours

    def render_fixed_num_var_dist_bezier(self, num_points):
        render_name = "fixed-num-var-diststraight-line-glyph"
        fixed_num_var_dist_contours = self.make_fixed_num_var_dist_bezier(num_points)        
        plt.figure()
        cl.render(fixed_num_var_dist_contours)
        for output_format in OUTPUT_FORMATS:
            file_name = "{}-{}-{}.{}".format(self.font_name, self.char_name, render_name, output_format)
            plt.savefig(os.path.join("outputs", "fixed-num-var-dist-straight-line", file_name))   
        plt.close()
    def make_fixed_num_point_contour(self, num_points):
        fixed_num_contours = []
        for contour in self.contours:

            fixed_num_contour = []
            start_points = []
            end_points = []

            contour_len = cl.get_contour_length(contour)
            dist_per_point = contour_len/(num_points)
            loc = 0
            len_sum = 0
            transposed_contour = np.transpose(contour, axes=(0, 2, 1)).astype(np.float64)

            first = True
            last = False

            for i in range(len(transposed_contour)):
                # check if the end point of current = start point of next. If not, flip the current one.
                bezier_curve = bezier.Curve(transposed_contour[i], degree = 2)
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
        return fixed_num_contours
    
    def render_fixed_num_distance_bezier(self, num_points):
        fixed_num_contours = self.make_fixed_num_point_contour(num_points)
        plt.figure()
        cl.render(fixed_num_contours)
        for output_format in OUTPUT_FORMATS:
            file_name = "{}-{}-fixed-straight-line-glyph.{}".format(self.font_name, self.char_name, output_format)
            plt.savefig(os.path.join("outputs", "fixed-straight-line", file_name))   
        plt.close()     
                    
if __name__ == "__main__":
    make_output_dirs()
    for glyph in args.glyph:
        glyph = Glyph(font_name = args.font, char_name=glyph)
        glyph.render_raw_glyph()
        glyph.render_straight_line_glyph()
        glyph.render_fixed_num_distance_bezier(num_points = args.points)
        glyph.render_fixed_num_var_dist_bezier(num_points = args.points)
