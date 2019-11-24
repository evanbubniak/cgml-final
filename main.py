from glaze import read_json

arial = read_json("Ariel.json")

A_glyph = arial[0][2]

A_straight_line_glyph = make_straight_line_glyph(A_glyph)

def make_straight_line_glyph(glyph):
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

# def fix_glyph

if __name__ == "__main__":
    # main script
    pass