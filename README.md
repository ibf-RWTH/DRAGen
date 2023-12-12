<h1> DRAGen - <ins>D</ins>iscrete <ins>R</ins>VE <ins>A</ins>utomation and <ins>Gen</ins>eration</h1>
<!--**D**iscrete **R**VE **A**utomation and **Gen**eration-->

<!--## Overview-->
<!--![logo](docs/GUI.PNG)-->

 [**Installation**](#Installation)
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

<p align="left"><img src="docs/DRAGen_readme_paper.jpg" height="400" alt=""> </img></p>

_Note: For developing it is highly recommended to use Python versions 3.6 to 3.8. For Python 3.9 Pyvista is not fully supported._<br>
**If further questions appear please check the lower section or get in touch with us.**


## Installation

As the first step, conda needs to be installed.
To be sure conda is installed correctly on your system [look up here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)<br>

Git must be installed on the system. Check with:
```
$ git --version
```
If it has not been installed use this:
```
$ conda install -c anaconda git
```
Open the user path and create the directory where the DRAGen repo will be cloned.
Should be like this:
```
(base) C:\Users> cd \Users\{username}
(base) C:\Users\username> mkdir GitRepos
(base) C:\Users\username> cd GitRepos
```
To clone this repository into the desired destination, use:<br>
```
$ git clone https://github.com/IMS-RWTH/DRAGen.git
```
To be able to use DRAGen, the working directory must be set to the location where the repo was downloaded to in the previous step file which is downloaded at the previous step.
Use the commands to go to the exact file by following the path.
```
$ cd DRAGen
```
To see the folders on the current point:
```
$ dir
```
Create a virtual environment as follows:<br>
```
$ conda create --name DRAGen python=3.8
$ conda activate DRAGen
```
(if an error occurs check your conda installation)<br>
To see the list of the environments on conda:
```
$ conda info --envs
```
Be sure the DRAGen environment is activated it should look somewhat like this:<br>
```
(DRAGen)....$ 
```
Install one of two required module packages depending on cuda availability on the device:

To install requirements without cuda:<br> 
```
(DRAGen)....$ pip install -r requirements.txt 
```
To install requirements if cuda is available:<br> 
```
(DRAGen)....$ pip install -r requirements_cuda.txt 
```
Check if every step is utilized correctly by running first generation with:<br>
```
(DRAGen)....$ python DRAGen_nogui.py
```
Run DRAGen:<br>
```
(DRAGen)....$ python DRAGen.py
```

## Related Projects

### MCRpy
<p align="center"><img src="docs/MCRpy-logo_png.png" height="200" alt="MCRpy logo"> </img></p>

[MCRpy](https://github.com/NEFM-TUDresden/MCRpy) (Microstructure Characterization and Reconstruction in Python) facilitates the process by employing a range of descriptors and enables the reconstruction of new microstructures. One key advantage of MCRpy is its extensibility, allowing the combination of various descriptors without the need for loss functions. Additionally, it provides flexibility in choosing optimizers to address emerging optimization problems.



### DAMASK
<p align="center"><img src="docs/DAMASK_banner.png" height="100" alt="DAMASK banner"> </img></p>

[DAMASK](https://damask.mpie.de/index.html) (Düsseldorf Advanced Materials Simulation Kit) excels in its ability to handle a variety of simulation programs under different conditions, particularly for advanced high-strength materials. Its capability to address the interconnected nature of deformation, phase transformations, heating effects, and potential damage makes DAMASK an invaluable choice for researchers and practitioners seeking a comprehensive understanding of materials behavior in diverse scenarios.





<details>
<summary><b>Show more...<b></summary>



## Input data

<!--* a: grain radius (**mandatory**)
* b: grain radius (optional, default = a )
* c: grain radius (optional, default = a)
* alpha: grain slope in x-y-plane (optional, default = 0)
* beta: grain slope in other plane (not yet implemented)
* phi1: euler angle (optional, default: random)
* PHI: euler angle (optional, default: random)
* phi2: euler angle (optional, default: random)<br>-->


| Header: | a | b | c | alpha | beta | phi1 | PHI | phi2 |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Description:** | grain radius | grain radius | grain radius | grain slope<br>x-y-plane | _soon_ | euler ang. | euler ang. | euler ang. |
| **Required:** | mandatory | optional | optional | optional | _soon_ | optional | optional | optional |
| **Default:** |  | a | a | 0 | _soon_ | random | random | random |

<br>
DRAGen takes .csv files as input. Theses files must contain <ins>at least one radius</ins> for each grain. This radius has to be called <em>a</em> in the header.
<br><ins>Optional parameters</ins> are:

1. _b_ and _c_ as second and third radius of each grain (ellipsoids are created).<br> _a_ is assumed to be oriented with the rolling direction and is aligned with x-axis,
_b_ is aligned with y-axis and _c_ with z-axis.<br>
2. If a slope relative to x-axis is detected (rotation in x-y-plane, around z-axis), _alpha_ can be used to implement this slope on the grains.<br>
_beta_ will be implemented in the future and will be a rotation around x- or y-axis.<br>
3. The texture can be defined with the parameters _phi1_, _PHI_ and _phi2_.

## Output Data

In V.1.0_b, the output files are Abaqus input files designed for the use with the ICAMS-Crystal-plysticity model. Therefore, the subroutine-files are needed for a successfull analysis.

* Periodic boundary conditions (PBC): BottomToTop.inp, FrontToRear.inp, LeftToRight.inp, Corners.inp, Edges.inp, Nsets.inp, VerticeSets.inp
* CP-model data (euler angles and grain size): graindata.inp
* RVE: RVE_smooth.inp
* RVE in arry: RVE_Numpy.npy (not needed at the moment!)

It is distiguished between a plastic phase (Phase 1, e.g. Ferrite) purely elastic phase (Phase 2, e.g. Martensite) and . Extensions to more then two phases are in the making.


## Input generator

One additional feature of our system is the generation of statistically representative microstructur using **Generative Adversarial Networks**, a method from the field of deep learning. With our CWGAN-GP, it is possible to generate an unlimited amount of vaild synthetical microstructure. Possible for "normal" grain data, inclusions and even damage (coming soon!). For more information, see our article on the basic idea of using a WGAN (https://www.mdpi.com/1996-1944/13/19/4236) and our IDDRG post on the CWGAN-GP (coming shortly after publishing).

## Latest Version
* DRAGen.V.1.0_b


## Support

* DRAGen Support <DRAGen@iehk.rwth-aachen.de>

Please use one of the following keywords for your issue as e-mail subject:
* General problems
* RSA error
* Tesselation error
* Mesher error
* Substructure
* Inclusions
* Bands


</details>
