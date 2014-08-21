README
======

A python code for fitting multi-color observations of planetary
transits. As well as providing a python module, I provide a flexible
and powerful python script for fitting multi-color transit data with
MCMC.

'rainbow' uses Ian Crossfield's Python implementation of the Mandel & Algol transit
model, which is included for convenience

INSTALLATION
------------

As well as "standard" Python dependencies (numpy, scipy, matplotlib), you will also need Dan's triangle package, for nice plots of posterior
distributions. You can install this with "pip install triangle-plot".

This is all you need to use the 'rainbow' module to calculate your own models (say if you want to write your own fitting code). The
recommended way of using this module is to run the 'transitModel.py' script. In this
case you will need the following additional dependencies:

 emcee : Models are fit to data files using MCMC, implemented using Dan
         Foreman-Mackey's excellent emcee package -
         http://dan.iel.fm/emcee/current. 

 george : Red noise is modelled using Gaussian
          Processes, again using Dan's package george -
          http://dan.iel.fm/george/current/. (Note: this is currently in 
          development).

Installation proceeds via the usual::

 python setup.py install
 
if you are root, or::

 python setup.py install --prefix=<install dir>
 
if you are not.

USAGE
-----

The installation will put two 

Operation of this script is controlled by a file named "input.dat". This
should be self-documenting. The example_data directory contains some
data to get you going.

Limb-darkening is handled using a quadratic limb-darkening law. The
utility script limbdark.py will help you find starting values in SDSS
filters, using the tables from Claret & Bloemen (2011).

