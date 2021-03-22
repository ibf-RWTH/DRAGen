# DRAGen

This repository includes the official implementation of the paper [A Novel Approach to Discrete Representative Volume Element Automation and Generation-DRAGen](https://www.mdpi.com/1996-1944/13/8/1887).

![logo](https://www.mdpi.com/materials/materials-13-01887/article_deploy/html/images/materials-13-01887-g003.png)

## Overview

DRAGen is an approach for generating Representative Volume Elements (RVEs) based on a Random Sequential Addition (RSA)-Algorithm for discrete volumes and the tessellation using a discrete tessellation function. The input data are gathered from the analysis of electron backscatter diffraction (EBSD) pictures via MATLAB toolbox MTEX and introduced to the model. Subsequently, the generator follows the below mentioned steps:

* Randomly generating ellipsoids in volume (RSA)
* Filling empty spaces between ellipsoids(Discrete Tessellation)
* Validation of the newly created digital microstructure with input data

The results show that the generator can successfully reconstruct realistic microstructures with elongated grains and martensite bands from given input data sets.

## Required Packages

The code is built upon:

- appdirs==1.4.4
- certifi==2020.6.20
- chardet==3.0.4
- click==7.1.2
- cycler==0.10.0
- idna==2.10
- imageio==2.9.0
- kiwisolver==1.2.0
- matplotlib==3.3.0
- meshio==4.3.11
- numexpr==2.7.1
- numpy==1.19.0
- pandas==1.0.5
- Pillow==7.2.0
- pyparsing==2.4.7
- PyQt5==5.15.0
- PyQt5-sip==12.8.0
- PyQt5-stubs==5.14.2.2
- pyqt5-tools==5.15.0.1.7
- pyqtgraph==0.11.0
- python-dateutil==2.8.1
- python-dotenv==0.14.0
- pytz==2020.1
- pyvista==0.29.0
- requests==2.24.0
- requests-cache==0.5.2
- scooby==0.5.6
- six==1.15.0
- tables==3.6.1
- tetgen==0.5.1
- tqdm==4.48.0
- transforms3d==0.3.1
- urllib3==1.25.10
- vtk==9.0.1

## Contact

* Manuel Henrich M. Sc. <manuel.henrich@iehk.rwth-aachen.de>
* Maximilian Neite M. Sc. <maximilian.neite@iehk.rwth-aachen.de>
* Niklas Fehlemann M. Sc. <niklas.fehlemann@rwth-aachen.de>
* Orkun Karagüllü M. Sc. <orkun.karaguellue@iehk.rwth-aachen.de>

