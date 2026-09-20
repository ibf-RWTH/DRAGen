import math
import logging
import typing
import numpy as np


class RveInfo:
    """
    This class stores all the constant values which are needed for the current RVE-Generation.
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
    box_size: any([float, int]) = None
    """box_size describes the edge length of the rve in a cubic case, if the rve is not cubic it describes the length
    of the edge in x-direction"""

    box_size_y: any([float, int, None]) = None
    """If this value is set to any other value than None the rve will take this value in µm and the 
    periodicity will be turned off in y-direction"""

    box_size_z: any([float, int, None]) = None
    """If this value is set to any other value than None the rve will take this value in µm and the 
    periodicity will be turned off in z-direction"""

    box_volume = float


    resolution: any([float, int]) = None
    """ number of elements along rve-edge = box_size*resolution: higher resolution --> finer mesh """

    low_rsa_resolution: bool = False
    """ flag if rsa should run with low resolution """

    allowed_intersection_ratio: float = 0.05
    """ accepted intersection ratio during rsa to speed up process Default 5%"""

    number_of_rves: int = None
    """choose a number of RVEs to be generated"""

    generation_time: float = None
    """overall generation time in seconds for the current RVE (set by Run.run() after each RVE completes)"""

    dimension: int = None
    """ choose as 2 or 3 for either a 2D or 3D RVE """

    slope_offset = None
    """angle of macro specimen relative to rolling direction for sheet material"""

    smoothing_flag: bool = None

    # martensite band parameters
    number_of_bands: int = None
    """chose a number of bands for each rve"""

    bandwidths: np.ndarray = None
    """np array containg the bandwidth of each band (calculated automatically)"""

    lower_band_bound: float = None
    """this parameter sets a lower limit for the band thickness"""

    upper_band_bound: float = None
    """this parameter sets a lower limit for the band thickness"""

    file_dict: dict = None
    """dictionary of all grain files"""

    phase_ratio: dict = None
    """dictionary of all phaseratios. Ratios must add up to one otherwise errors will occur"""

    store_path: str = None
    """output path for generation files"""

    band_ratio_rsa: float = None

    band_ratio_final: float = None

    band_filling: float = None
    """Percentage of Filling for the banded structure/ similar to shrinkfactor just for the band"""

    band_orientation: str = None
    """Define the plane in which the band lies: 'xy', 'yz' or 'zy'"""

    gui_flag: bool = None
    """set to False when using the nogui_script"""

    infobox_obj = None
    """internal object for GUI"""
    progress_obj = None
    """internal object for GUI"""

    fig_path: str = None
    gen_path: str = None
    post_path: str = None

    # substructure generation parameters
    num_cores: int = 1  # if > 1, multiprocessing used
    t_mu: float = 1.0
    """mean block thickness in micrometres, used when subs_file_flag is False"""
    subs_file_flag: bool = None
    """if True, block thicknesses are sampled from subs_file instead of from t_mu"""
    subs_file: str = None
    """csv of measured block thicknesses; needs a 'block_thickness' column"""
    subs_flag: bool = None

    # substructure pipeline parameters (dragen/substructure/), mirrored into SubsConfig
    subs_transformable_phase_ids: list = None
    """phase ids that get packets/blocks; everything else stays unsubstructured. Default [2, 3, 4]"""
    subs_lower_percentile: float = 5.0
    subs_upper_percentile: float = 95.0
    """percentile clip applied to the measured block thickness distribution (subs_file)"""
    subs_min_packet_cells: int = 100
    """target cells per packet for the k-means split of an interior grain"""
    subs_min_block_cells: int = 10
    """packets with fewer cells than this are kept as a single block instead of being sliced"""
    subs_min_cells_per_packet: int = 5
    subs_min_cells_per_block: int = 5
    """packets/blocks below this size are merged into a neighbour"""
    subs_orientation_mode: str = 'KS'
    """'KS' for the 24 ideal Kurdjumov-Sachs variants, 'experimental' for a measured OR"""
    subs_parent_orientation_file: str = None
    """csv of parent (PAG) orientations; defaults to the phase input of the first transformable phase"""
    subs_child_orientation_file: str = None
    """csv of measured block orientations with grain_id, required for 'experimental' mode"""

    phases: list = None
    """List of phase names"""

    abaqus_flag: bool = None
    """Set to True for Abaqus input file"""
    subroutinetype: dict = {'ICAMS': True, 'TRIP': False}

    damask_flag: bool = None
    """Set to True for DAMASK input file"""

    moose_flag: bool = None
    """Set to True for MOOSE input file"""

    calibration_rve_flag: bool = None
    """An RVE with 1000 elements is generated each element is considered one grain
       the Orientation is chosen randomly and the phase ratio is taken from the input
       the regular generation process with rsa and tessellation is ignored for this type
       of RVE
    """

    debug: bool = True
    """In debug mode, the validity of passed parameters and returned results will be checked to ensure properly-set 
        parameters. But this can slow down the running. So after debugging, it can be turned off to speed running up.
    """

    anim_flag: bool = None
    """If set to True RSA and Tesselation will be plotted"""

    phase2iso_flag: dict = None
    """assumption for abaqus that second phase is treated as isotropic material"""

    pbc_flag: bool = None
    """If set to True periodic boundary conditions will be applied ( if True submodel_flag must be False!!)"""

    submodel_flag: bool = False
    """If set to True a submodel usage will be assumed (if True pbc_flag must be False!!!)"""
    xfem_flag: bool = False

    element_type: str = None
    """available element types: (C3D4, HEX8)"""
    reduced_elements: bool = True

    roughness_flag: bool = False
    """Flag not yet activated since roughness is not yet implemented"""

    texture_flag: bool = False
    """Flag to activate Texture analysis if texture is needed"""

    root: str = './'
    """root path"""

    PHASENUM = {'Ferrite': 1, 'Martensite': 2, 'Pearlite': 3, 'Bainite': 4, 'Austenite': 5, 'Inclusions': 6, 'Bands': 7}
    """Numbers linked to currently defined phases"""

    rwth_colors = [(100 / 256, 101 / 256, 103 / 256),  # grey
                   (0 / 256, 83 / 256, 159 / 256),  # blue
                   (204 / 256, 7 / 256, 30 / 256),  # red
                   (142 / 256, 186 / 256, 229 / 256),  # light blue
                   (246 / 256, 168 / 256, 0 / 256),  # orange
                   (161 / 256, 16 / 256, 53 / 256)]  # Bordeaux
    """All graphs are currently plotted in RWTH-Color Coding"""

    n_pts = None
    """Number of points on edge along x-axis. Value is automatically calculated"""

    n_pts_y = None
    """Number of points on edge along y-axis. Value is automatically calculated"""

    n_pts_z = None
    """Number of points on edge along z-axis. Value is automatically calculated"""

    bin_size = None
    """element size one point in grid represents"""

    LOGGER = logging.getLogger("RVE-Gen")
    RESULT_LOG = logging.getLogger("RVE-Result")

    ######### Fixed Constants can only be changed here #########

    SHRINK_FACTOR: float = np.cbrt(0.3)
    """factor by which all ellipsoids are shrinked before beeing placed in the volume"""
