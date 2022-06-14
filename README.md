<h1> DRAGen - <ins>D</ins>iscrete <ins>R</ins>VE <ins>A</ins>utomation and <ins>Gen</ins>eration</h1>
<!--**D**iscrete **R**VE **A**utomation and **Gen**eration-->

<!--## Overview-->
<!--![logo](docs/GUI.PNG)-->

This repository includes the official implementation of the paper [A Novel Approach to Discrete Representative Volume Element Automation and Generation-DRAGen](https://www.mdpi.com/1996-1944/13/8/1887). 
DRAGen is an approach for generating Representative Volume Elements (RVEs) based on a Random Sequential Addition (RSA)-Algorithm for discrete volumes and the tessellation using a discrete tessellation function. The input data are gathered from the analysis of electron backscatter diffraction (EBSD) pictures via MATLAB toolbox MTEX and introduced to the model. Subsequently, the generator follows the below mentioned steps:

* Randomly generating ellipsoids in volume (RSA)
* Filling empty spaces between ellipsoids (Discrete Tessellation)
* Validation of the newly created digital microstructure with input data

The results show that the generator can successfully reconstruct realistic microstructures with elongated grains and martensite bands from given input data sets.

![logo](docs/GUI_Banding.PNG)

_Note: For developing it is highly recommended to use Python versions 3.6 to 3.8. For Python 3.9 Pyvista is not fully supported._<br>
**If further questions appear please check the lower section or get in touch with us.**


## Installation
clone this repository into desired destination:<br>
```
$ git clone https://git.rwth-aachen.de/ims_iehk/dragen.git
$ cd dragen
```
make sure conda is installed correctly on your system [look up here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)<br>
Create a virtual environment as follows:<br>
```
$ conda create --name DRAGen python=3.8
$ conda activate DRAGen
```
(if an error occurs check your conda installation)<br>
make sure the DRAGen env is activated it should look somewhat like this:<br>
```
(DRAGen)....$ 
```
install requirements:<br>
```
(DRAGen)....$ pip install -r requierments.txt
```
run first generation with:<br>
```
(DRAGen)....$ python DRAGen_nogui.py
```


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
