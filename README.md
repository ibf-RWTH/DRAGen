<h1> DRAGen - <ins>D</ins>iscrete <ins>R</ins>VE <ins>A</ins>utomation and <ins>Gen</ins>eration</h1>
<!--**D**iscrete **R**VE **A**utomation and **Gen**eration-->

<!--## Overview-->
<!--![logo](docs/GUI.PNG)-->

 [**Installation**](#Installation)
| [**Use as a library**](#use-as-a-library)
| [**Related Projects**](#Related-Projects)
| [**Tutorial**](Tutorial)


This repository presents an enhanced version of the Discrete Representative Volume Element (RVE) Automation and Generation Framework, known as DRAGen. Originally devised as an approach for generating Representative Volume Elements based on a Random Sequential Addition (RSA)-Algorithm and discrete tessellation, DRAGen has undergone significant improvements to broaden its capabilities. DRAGen collects and processes data from analysis of electron backscatter diffraction (EBSD) images via the [MATLAB toolbox MTEX](https://mtex-toolbox.github.io/)  or [OIM](https://www.edax.com/products/ebsd/oim-analysis) and presents it as a model. Subsequently, the generator follows the below mentioned steps:

* Randomly generating ellipsoids in volume (RSA)
* Filling empty spaces between ellipsoids (Discrete Tessellation)
* Validation of the newly created digital microstructure with input data

The updated framework incorporates a generator for RVEs with several advanced features, drawn from real microstructures. DRAGen now possesses the ability to read input data from trained neural networks and .csv files, offering greater flexibility in microstructure generation. Notably, the generator has been enriched to reconstruct microstructures with intricate features such as;

* Pores
* Inclusions 
* Martensite bands 
* Hierarchical substructures
* Crystallographic textures.

In addition to these enhancements, DRAGen has been extended to support different solvers. DRAGen is capable of creating models compatible with three widely used multiphysics frameworks: [DAMASK](https://damask.mpie.de/index.html), [Abaqus](https://www.3ds.com/products-services/simulia/products/abaqus/), and [MOOSE](https://mooseframework.inl.gov/).

Its versatility makes it a valuable tool for scientists in the Integrated Computational Materials Engineering (ICME) community. The modular architecture of the project facilitates easy expansion with additional features, ensuring that DRAGen delivers a diverse range of functions and outputs. This diversity offers a comprehensive spectrum of microstructures, thereby contributing to the advancement of microstructure studies and the development of innovative microstructure designs.

For more:

[A Novel Approach to Discrete Representative Volume Element Automation and Generation-DRAGen](https://www.mdpi.com/1996-1944/13/8/1887)

[DRAGen – A deep learning supported RVE generator framework for complex microstructure models](https://www.sciencedirect.com/science/article/pii/S2405844023062114#fg0340)

[Generating Input Data for Microstructure Modelling: A Deep Learning Approach Using Generative Adversarial Networks](https://www.mdpi.com/1996-1944/13/19/4236)

[Identification of martensite bands in dual‐phase steels: a deep learning object detection approach using faster region‐based‐convolutional neural network](https://onlinelibrary.wiley.com/doi/pdf/10.1002/srin.202200836)

[Exploring structure–property relations in dual phase steels using crystal plasticity and variance based global sensitivity analysis](https://www.sciencedirect.com/science/article/pii/S0264127525012146)

[Crystal plasticity parameter identification via statistical relevant micropillar compression](https://www.sciencedirect.com/science/article/pii/S135964542500607X#cited-by)

<p align="left"><img src="docs/DRAGen_readme_paper.jpg" height="400" alt=""> </img></p>

**If further questions appear please check the lower section or get in touch with us.**


## Installation

DRAGen needs Python 3.12 or newer. Clone the repository and install it into a virtual environment
(conda, venv or uv):

```
git clone https://github.com/ibf-RWTH/DRAGen.git
cd DRAGen
python -m venv .venv                      # or: conda create --name DRAGen python=3.12
.venv/Scripts/activate                    # Linux/macOS: source .venv/bin/activate
pip install -e ".[ml,damask]"             # ml: GAN grain sources (torch), damask: DAMASK export
```

Check the installation with a first RVE:

```
dragen template config.yaml               # example configuration (micrometres)
dragen generate config.yaml -o out -f grains,vtu,abaqus
dragen gui                                # browser front end "DRAGen Studio"
```

`dragen generate` writes one directory per RVE (`out/rve_000`, ...) with `rve.npz`,
`metadata.json`, the configuration and the requested exports (`-f damask,abaqus,exodus,vtu,gmsh,
grains` or `all`; `--mesh tet` adds a tetrahedral mesh with smoothed grain boundaries). A
`substructure:` section in the configuration adds martensite packets and blocks
(`dragen.substructure.SubstructureConfig`). `dragen export out/rve_000 -f damask` exports a saved
RVE again. The PyQt GUI and the generator of DRAGen 1.x (`DRAGen.py`, `DRAGen_nogui.py`) are
available on the git tag `legacy-baseline`.

## Use as a library

DRAGen can be called from Python without a GUI through `dragen.api`. The core needs only numpy,
scipy, pandas, pyyaml, gmsh and meshio; everything else is an optional extra:

| Extra | For |
|---|---|
| `ml` | grain sources from trained GANs (`.pkl`): torch, geomloss, qhoptim, matplotlib (Agg), seaborn, tqdm |
| `damask` | `export_damask` |
| `test` | pytest, netCDF4 (reads the Exodus files back in the tests) |

```
pip install -e ".[ml,damask]"          # or, from another uv project: uv add --editable "../DRAGen[ml,damask]"
```

Example (run from the repository root, it uses the trained GANs in `ExampleInput`):

```python
from pathlib import Path
from dragen.api import PhaseSpec, RVESpec, generate_rve
from dragen.api import BandSpec, InclusionSpec        # optional features
from dragen.export import export_abaqus, export_damask, export_exodus, export_grains, export_vtu

spec = RVESpec(
    box_size_m=5.5e-6,              # cube edge in metres (or (x, y, z))
    resolution=16,                  # voxels along x
    phases=[PhaseSpec('ferrite', 0.7, Path('ExampleInput/Ferrite/TrainedData_Ferrite.pkl')),
            PhaseSpec('martensite', 0.3, Path('ExampleInput/Martensite/TrainedData_Martensite.pkl'))],
    mesh='tet',                     # 'voxel' (default) or also a tetrahedral mesh
    seed=1,
)
rve = generate_rve(spec)
print(rve.grain_ids.shape, rve.orientations.shape, rve.tet_mesh.cells.shape)
export_damask(rve, Path('out/damask'))  # grid.vti + material.yaml (no load case)
export_abaqus(rve, Path('out/rve'))     # rve.inp (mesh, sets, sections, PBC) + graindata.inp
export_exodus(rve, Path('out/rve'))     # rve.e for MOOSE: block per grain, box-face sets
export_vtu(rve, Path('out/rve'))        # rve.vtu; also export_gmsh
export_grains(rve, Path('out/grains'))  # grains.csv, one row per grain
```

2D: `RVESpec(box_size_m=(x, y), ...)` gives a pixel RVE (`grain_ids` of shape (nx, ny)); with
`mesh='tet'` it gets a triangle mesh (3 or 6 nodes) with smoothed grain boundaries, and the
exports write plane elements (Abaqus CPE4/CPE3/CPE6, Exodus QUAD4/TRI3/TRI6, DAMASK one cell layer).

Martensite substructure: `from dragen.substructure import SubstructureConfig, generate_substructure,
apply_substructure`; `apply_substructure(rve, generate_substructure(rve, SubstructureConfig(
block_thickness=1.0)))` returns an RVE with one region per block (3D).

Bands and inclusions: `RVESpec(..., bands=BandSpec(count=1, thickness_m=(3e-6, 4e-6),
source=..., phase='martensite'), inclusions=InclusionSpec(source=..., fraction=0.01))`. The grains
of a band are merged into one grain (`rve.band_ids` gives the band of a grain).

Exports: the mesh exports (Abaqus, Exodus, VTU, MSH) take the tetrahedral mesh if there is one,
otherwise one hexahedron per voxel (`mesh='voxel'|'tet'` chooses). The Abaqus file is model data
without materials and steps: include it in a job file that defines one material per section
(`Ferrite_12`, ... or per phase with `material_per='phase'`) and loads the reference nodes
`RP_X/RP_Y/RP_Z`, whose displacement is the jump across the periodic box. Exodus side and node sets
carry the names of MOOSE's GeneratedMesh (`left/right`, `bottom/top`, `back/front`).

Conventions (details in the docstring of `dragen/api.py`):

* Lengths in metres; the grain sources (GAN, CSV) stay in micrometres.
* Axes x = RD, y = TD, z = ND; `grain_ids[i, j, k]` is the voxel at (x, y, z).
* Grain IDs 1 … G; row g − 1 of `orientations` and `phase_ids` belongs to grain g
  (`rve.orientation_of(ids)` does the lookup).
* Orientations as unit quaternions (w, x, y, z), w ≥ 0, active rotation crystal → sample. These are
  the same four numbers DAMASK uses for the orientation; `euler_bunge_deg` holds the Bunge angles.
* Orientations come from the grain source if it has `phi1, PHI, phi2`, from
  `PhaseSpec(orientations=...)` (a CSV of measured orientations) if given, otherwise they are random.
* Same spec, same input files and same seed give the same RVE.

The library writes nothing to stdout; messages go to `logging`. `generate_rve(spec, out_dir)` also
saves `rve.npz` and `metadata.json` (version, git commit, seed, SHA-256 of spec and input files,
run times, warnings).

## Related Projects

### MCRpy
<p align="center"><img src="docs/MCRpy-logo_png.png" height="200" alt="MCRpy logo"> </img></p>

[MCRpy](https://github.com/NEFM-TUDresden/MCRpy) (Microstructure Characterization and Reconstruction in Python) facilitates the process by employing a range of descriptors and enables the reconstruction of new microstructures. One key advantage of MCRpy is its extensibility, allowing the combination of various descriptors without the need for loss functions. Additionally, it provides flexibility in choosing optimizers to address emerging optimization problems.



### DAMASK
<p align="center"><img src="docs/DAMASK_banner.png" height="100" alt="DAMASK banner"> </img></p>

[DAMASK](https://damask.mpie.de/index.html) (Düsseldorf Advanced Materials Simulation Kit) excels in its ability to handle a variety of simulation programs under different conditions, particularly for advanced high-strength materials. Its capability to address the interconnected nature of deformation, phase transformations, heating effects, and potential damage makes DAMASK an invaluable choice for researchers and practitioners seeking a comprehensive understanding of materials behavior in diverse scenarios.


### Moose 

<p align="center"><img src="docs/moose_logo.png" height="80" alt="Moose banner"> </img></p>

[MOOSE](https://mooseframework.inl.gov/) (Multiphysics Object-Oriented Simulation Environment) framework is an open-source software framework designed for solving complex physics-based engineering simulations. Moose supports adaptive mesh refinement which improves the accuracy and efficiency. It is possible to develop a simulation application tailored to specific needs. 


### MTEX

<p align="center"><img src="docs/MTEX_100x67px.png" height="130" alt="MTEX banner"> </img></p>

[MTEX](https://mtex-toolbox.github.io/index) (Materials Texture and Microstructure Analysis Toolbox) is an open-source Matlab tool that allows data processing using techniques such as EDSB, and analysis on microstructure and crystallographic textures. It has powerful visualization tools for displaying crystallographic data, including pole figures, stereographic projections, and other graphical representations of orientation information.



