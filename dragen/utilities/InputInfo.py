import math
import logging
import typing
import numpy as np

class RveInfo:
    """
    This class simply stores all the constant values which are needed for the current RVE-Generation.
    If one of theses Parameters is needed in any of the following classes/modules/functions for generating the RVE
    it can be called by importing this class as follows:
    from InputInfo import RveInfo
    and the constants can be accessd as follows:
    box_size = RveInfo.box_size
    .
    .
    .
    WARNING!!! BE VERY CAREFUL WHEN OVERWRITING THESE VALUES OUTSIDE OF THE dragen\run.py!!!!!
    """
    box_size: any([float, int]) = 20  #
    """box_size describes the edge length of the rve in a cubic case, if the rve is not cubic it describes the length
    of the edge in x-direction"""
    box_size_y: any([float, int]) = None
    box_size_z: any([float, int]) = None
    resolution: any([float, int]) = 1
    number_of_rves: int = 1
    number_of_bands: int = 0
    bandwidths: np.ndarray = None
    lower_band_bound: float = None
    upper_band_bound: float = None
    dimension: int = 3
    visualization_flag: bool = False
    file_dict: dict() = None
    phase_ratio: float = None  # 1 means all ferrite, 0 means all Martensite TODO: Use a dict
    store_path: str = None
    shrink_factor: float = None
    band_ratio_rsa: float = None
    band_ratio_final: float = None
    gui_flag: bool = None
    gan_flag: bool = None
    infobox_obj = None
    progress_obj = None
    equiv_d: float = None
    p_sigma: float = None
    t_mu: float = None
    b_sigma: float = None
    decreasing_factor: float = None
    lower: float = None
    upper: float = None
    circularity: float = None
    plt_name: str = None
    save = None
    plot = None
    filename: str = None
    fig_path: str = None
    gen_path: str = None

    orientation_relationship: str = None
    subs_file_flag: bool = None
    subs_file: bool = None
    phases: list = None
    subs_flag: bool = None
    abaqus_flag: bool = None
    damask_flag: bool = None
    moose_flag: bool = None
    debug: bool = False
    sub_run = None
    anim_flag: bool = None
    exe_flag: bool = None
    phase2iso_flag: bool = None
    element_type: str = None
    roughness_flag: bool = False
    band_filling: float = 0.99  # Percentage of Filling for the banded structure
    inclusion_ratio: float = 0.01  # Space occupied by inclusions
    inclusion_flag: bool = None
    root: str = './'
    input_path: str = './ExampleInput'
    PHASENUM = {'ferrite': 1, 'martensite': 2, 'Pearlite': 3, 'Bainite': 4, 'Inclusions': 5, 'Bands': 6}


    n_pts = None
    n_pts_y = None
    n_pts_z = None
    bin_size = None
    step_half = None

    logger = logging.getLogger("RVE-Gen")
    result_log = logging.getLogger("RVE-Result")
