README
======

A python code for fitting multi-color observations of planetary
transits. As well as providing a python module, I provide a flexible
and powerful python script for fitting multi-color transit data with
MCMC. Using multiple threads is supported for speed.

'rainbow' uses Ian Crossfield's Python implementation of the Mandel & Algol transit
model, which is included for convenience

INSTALLATION
------------

As well as "standard" Python dependencies (numpy, scipy, matplotlib), you will also need Dan's triangle package, for nice plots of posterior
distributions. You can install this with "pip install triangle-plot". You may also need python-tk, if that is not
installed by default.

This is all you need to use the 'rainbow' module to calculate your own models (say if you want to write your own fitting code). The
recommended way of using this module is to run the 'transitModel.py' script. In this
case you will need the following additional dependencies:

 emcee : Models are fit to data files using MCMC, implemented using Dan Foreman-Mackey's excellent emcee package - http://dan.iel.fm/emcee/current. 

 george : Red noise is modelled using Gaussian Processes, again using Dan's package george - http://dan.iel.fm/george/current/. I recommend using the version installed via 'pip install george'. 

Installation proceeds via the usual::

 python setup.py install
 
if you are root, or::

 python setup.py install --prefix=<install dir>
 
if you are not.

USAGE
-----

The installation will put two utility scripts in your path. The main script
'transitModel.py' takes no arguments by default.  Running transitModel.py
is the recommended way of using this module. You should be able to fit
any arbitrary multi-colour dataset using this script, without delving into 
more advanced usage patterns.

Operation of this script is controlled by a file named "input.dat". This
should be self-documenting! The example_data directory contains some
data and an input.dat file to get you going.

Upon completion, the script shows a plot of the model and the data. The 
upper panel displays the model fit and the data *with the GP noise model
subtracted*. The bottom panel shows the residuals between the model and
the *raw data*. Over-plotted on the bottom panel is the 1-sigma range of noise
models arising from the Gaussian process.

Limb-darkening is handled using a quadratic limb-darkening law. The
utility script limbdark.py will help you find initial values for SDSS
filters, using the tables from Claret & Bloemen (2011).

KNOWN_ISSUES
-------------

If you are using Mac OS X, and the Ureka python installation, you cannot
fit data with GP red noise and use multiple threads.


ADVANCED USAGE
--------------

Should you wish to write your own fitting procedures, you can do so as
follows. First import the rainbow module, and the Param and Prior objects from the 
utility module::

 import rainbow as r
 from rainbow.utils.mcmc_utils import Param, Prior
 
Param and Prior objects are used to supply parameters to the lightcurve model. Prior objects are initialised with a type (which can be one of 'gauss', 'gaussPos', 
'uniform' and 'log_uniform'), and a range. Gaussian-type priors are created like::

 prior = Prior('gauss',mean,stdDev)
 
whereas uniform-type priors are created with::

 prior = Prior('uniform',low_limit,high_limit)
 
Param objects are created with a starting value and a prior, like so::

 par = Param(0.1,prior)
  
A model for one colour can be initialised with the orbital period (which is not fit)
and parameters for mid-transit time, t0, impact parameter, b, stellar radius (scaled
to the separation, rs_a, the planetary radius (scaled to the stellar radius (rp_rs), the out-of-transit flux, f0, limb darkening parameters u1, u2
and two parameters which define an airmass term (see Copperwheat et al 2012 for details)::

 per   = 1.091423
 t0    = Param(54835.904, Prior('uniform',0,1))
 b     = Param(0.396,     Prior('uniform',0,1))
 rs_a  = Param(0.337,     Prior('uniform',0,1))
 rp_rs = Param(0.119,     Prior('uniform',0,1))
 f0    = Param(1.0,       Prior('uniform',0.1,10))
 u1    = Param(0.31,      Prior('uniform',0,1))
 u2    = Param(0.325,     Prior('uniform',0,1))
 A     = Param(-0.001,    Prior('uniform',-1,1))
 B     = Param(0.0017,    Prior('uniform',-1,1))
 
 model = rainbow.TransitModel(per,t0,b,rs_a,rp_rs,f0,u1,u2,A,B)

additional bands can be added to the model with the 'addBand' member::

 model.addBand(rp_rs, f0, u1, u2, A, B)

obviously we do not need to specify the stellar radius, impact parameter or mid-transit
time again, since these are shared between bands. This model does not include and GP noise.

Data from multiple bands should be stored in python lists, as follows::

 x = []
 y = []
 e = []
 # add red data
 x.append( x_red )
 y.append( y_red )
 e.append( e_red )
 # add grn data
 x.append( x_grn )
 y.append( y_grn )
 e.append( e_grn )

You can then evalulate the fit of the model to the data, either using chi-square,
the likelihood, or the posterior and prior probabilities::

 print 'For this model:'
 print "Reduced chisq  =  %.2f (%d D.O.F)" % (model.reducedChisq(x,y,e),np.size(x) - model.npars - 1)
 print "Chisq          = %.2f" % model.chisq(x,y,e)
 print "ln likelihood  = %.2f" % model.ln_likelihood(x,y,e)
 print "ln probability = %.2f" % model.lnprob(x,y,e)
 print "ln prior       = %.2f" % model.ln_prior()

Updating the model parameters is either done by accessing the parameters directly, or by
setting from a list of parameters. The current list of parameters can also be obtained from the 
model itself::

 model.t0.currval = 54835.86
 model.rp_rs[1].currVal = 0.12
 currPars = [par for par in model]
 model[0] = currPars[0] + 0.001 # increasing t0

Finally, the model (and the airmass term) can be calculated at a range of x positions::

 xmin = model.t0 - 0.2*per
 xmax = model.t0 + 0.2*per
 x = np.linspace(xmin,xmax,1000)
 band = 0 # calculate first colour
 y = model.calc(band,x)
 y_am = model.calc_airmass_term(band,x)
 
If you want to do your own fitting, but want a model which includes Gaussian Process noise, 
use the TransitModelGP class. Examples of use can be seen in the transitModel.py fitting 
script.
 
 