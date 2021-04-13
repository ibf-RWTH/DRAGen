# DRAGen

This repository includes the official implementation of the paper [A Novel Approach to Discrete Representative Volume Element Automation and Generation-DRAGen](https://www.mdpi.com/1996-1944/13/8/1887). It is highly recommended to use Python versions 3.6 to 3.8. For Python 3.9 there are still some problems with PyVista.

![logo](https://www.mdpi.com/materials/materials-13-01887/article_deploy/html/images/materials-13-01887-g003.png)

## Overview

DRAGen is an approach for generating Representative Volume Elements (RVEs) based on a Random Sequential Addition (RSA)-Algorithm for discrete volumes and the tessellation using a discrete tessellation function. The input data are gathered from the analysis of electron backscatter diffraction (EBSD) pictures via MATLAB toolbox MTEX and introduced to the model. Subsequently, the generator follows the below mentioned steps:

* Randomly generating ellipsoids in volume (RSA)
* Filling empty spaces between ellipsoids(Discrete Tessellation)
* Validation of the newly created digital microstructure with input data

The results show that the generator can successfully reconstruct realistic microstructures with elongated grains and martensite bands from given input data sets.

## Input generator

One additional feature of our system is the generation of statistically representative microstructur using **Generative Adversarial Networks**, a method from the field of deep learning. With our CWGAN-GP, it is possible to generate an unlimited amount of vaild synthetical microstructure. Possible for "normal" grain data, inclusions and even damage (coming soon!). For more information, see our article on the basic idea of using a WGAN (https://www.mdpi.com/1996-1944/13/19/4236) and our IDDRG post on the CWGAN-GP (coming shortly after publishing).

## soon to come
Release of beta v.1.0

## Contact

* Manuel Henrich M. Sc. <manuel.henrich@iehk.rwth-aachen.de>
* Maximilian Neite M. Sc. <maximilian.neite@iehk.rwth-aachen.de>
* Niklas Fehlemann M. Sc. <niklas.fehlemann@iehk.rwth-aachen.de>
* Orkun Karagüllü M. Sc. <orkun.karaguellue@iehk.rwth-aachen.de>

