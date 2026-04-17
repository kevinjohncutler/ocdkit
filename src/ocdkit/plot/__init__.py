"""Plotting utilities — figure creation, image grids, label styling, and colorization."""

from .figure import figure, split_list
from .grid import image_grid, color_swatches
from .contour import vector_contours
from .ncolor import apply_ncolor, sinebow
from .color import colorize, rgb_flow, rgb_to_hsv, hsv_to_rgb
from .label import recolor_label, add_label_background, apply_label_backgrounds
from .defaults import apply_mpl_defaults, setup
from .display import imshow, set_outline
from .export import export_gif, export_movie
