#!/usr/bin/env python

# fit exoplanet transit with model using MCMC
# 
import numpy as np
import emcee
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from rainbow import *
from rainbow.utils.mcmc_utils import *
import sys

import argparse
parser = argparse.ArgumentParser(description='Fit or plot model of transit in multiple bands')
parser.add_argument('--bins','-b',action='store',type=int,default=0,help='number of bins (0 for no binning)')
args = parser.parse_args()
nbins = args.bins

input_dict = parseInput('input.dat')	

nburn    = int( input_dict['nburn'] )
nprod    = int( input_dict['nprod'] )
nthread  = int( input_dict['nthread'] )
nwalkers = int( input_dict['nwalkers'] )
scatter  = float( input_dict['scatter'] )
toFit    = int( input_dict['fit'] )

ncolours = int( input_dict['ncolours'] )
hjdOff   = float( input_dict['hjdOff'] )
per      = float( input_dict['per'] )
t0 = parseParam(input_dict['t0'])
b = parseParam(input_dict['b'])
rs_a = parseParam(input_dict['rs_a'])
kernel = input_dict['kernel']
tau = parseParam(input_dict['tau'])

files = []
rp    = []
f0    = []
u1    = []
u2    = []
A     = []
B     = []
rn_amp = []
for col in range(1,1+ncolours):
    files.append( input_dict['file_%d' % col] )
    rp.append( parseParam( input_dict['rp_rs_%d' % col] ) )
    f0.append( parseParam( input_dict['f0_%d' % col] ) )
    u1.append( parseParam( input_dict['u1_%d' % col] ) )
    u2.append( parseParam( input_dict['u2_%d' % col] ) )
    A.append(  parseParam( input_dict['A_%d' % col] ) )
    B.append(  parseParam( input_dict['B_%d' % col] ) )
    rn_amp.append( parseParam( input_dict['rn_amp_%d' % col] ) )

# create a transit model from the first band's parameters
model = TransitModelGP(per,t0,b,rs_a,kernel,tau,rp[0],f0[0],u1[0],u2[0],A[0],B[0],rn_amp[0])

# then add additional colours with the models addBand function
for col in range(1,ncolours):
    model.addBand(rp[col],f0[col],u1[col],u2[col],A[col],B[col],rn_amp[col])

# store your data in python lists, so that x[0] are the times for colour 0, etc.
x = []
y = []
e = []
for file in files:
    data = np.loadtxt(file)
    x.append( data[:,0]-hjdOff )
    y.append( data[:,1] )
    e.append( data[:,2] )

npars= model.npars
params = [par for par in model]

def lnprob(pars,model,x,y,e):
    # we need to update the model we're using to use pars as submitted by MCMC
    for i in range(model.npars):
        model[i] = pars[i]
    return model.lnprob(x,y,e)

if toFit:
    p0 = np.array(params)
 
    # scatter values around starting guess
    p0 = emcee.utils.sample_ball(p0,scatter*p0,size=nwalkers)
    sampler = emcee.EnsembleSampler(nwalkers,npars,lnprob,args=[model,x,y,e],threads=nthread)

    #burn-in
    print 'Running burn-in'
    pos, prob, state = run_burnin(sampler,p0,nburn)

    #production
    print 'Running main chain'
    sampler.reset()
    sampler = run_mcmc_save(sampler,pos,nprod,state,"chain.dat")
    chain   = flatchain(sampler.chain,npars,thin=1)
    
    nameList = ['T0','b','Rs/a','tau']
    for icol in range(ncolours):
        nameList.extend( ['Rp/Rs %d' % (icol+1), \
            'Fs %d' % (icol+1), \
            'u2 %d' % (icol+1), \
            'A %d' % (icol+1), \
            'B %d' % (icol+1), \
            'rn %d' % (icol+1)] )
    thumbPlot(chain,nameList)

    params = []
    for i in range(npars):
        par = chain[:,i]
        lolim,best,uplim = np.percentile(par,[16,50,84])
        print "%s = %f +%f -%f" % (nameList[i],best,uplim-best,best-lolim)
        model[i] = best
        params.append(best)

#update model with best fit
for i, par in enumerate(params):
    model[i] = par
    
print 'For this model:'
print "Reduced chisq  =  %.2f (%d D.O.F)" % (model.reducedChisq(x,y,e),np.size(x) - model.npars - 1)
print "Chisq          = %.2f" % model.chisq(x,y,e)
print "ln probability = %.2f" % model.lnprob(x,y,e)
print "ln prior       = %.2f" % model.ln_prior()
plotColours = ['r','g','b']
gs = gridspec.GridSpec(2,ncolours,height_ratios=[2,1])
gs.update(hspace=0.0)

# make all y plots share y-axis with first 
minY = 1e32
maxY = -1e32
LHplot = True
for icol in range(ncolours):

    # rebin if required
    if nbins >= 1:
        bins = np.linspace(x[icol].min(),x[icol].max(),nbins)
        xp,yp,ep = rebin(bins,x[icol],y[icol],e[icol],weighted=True,errors_from_rms=False)
    else:
        xp = x[icol].copy()
        yp = y[icol].copy()
        ep = e[icol].copy()

    fy = model.calc(icol,xp)
    am = model.calc_airmass_term(icol,xp)
    samples = model.sample_conditional(icol,xp,yp,ep)
    mu = np.mean(samples,axis=0)
    std = np.std(samples,axis=0)
        
    # remove airmass term from plots
    #fy /= am
    #yp /= am
    #ep /= am
    
    # phase fold
    xp = (xp-model.t0.currVal)/ per
    
    ynorm = fy.max()
    
    ax1 = plt.subplot(gs[0,icol])	
        
    ax1.errorbar(xp,(yp-mu)/ynorm,yerr=ep/ynorm,color=plotColours[icol],fmt='.')
    ax1.plot(xp,fy/ynorm,'k-',linewidth=3.)
    ax1.set_xticks([])
    
    lo,hi = ax1.get_ylim()
    minY = min(lo,minY)
    maxY = max(hi,maxY)
        
    ax2 = plt.subplot(gs[1,icol],sharex=ax1)		
    ax2.errorbar(xp,(yp-fy)/ynorm,yerr=ep/ynorm,color=plotColours[icol],fmt='.',alpha=0.7)
    ax2.fill_between(xp,mu+std,mu-std,color='k',alpha=0.7)
    ax2.set_xlim(ax1.get_xlim())

    #labels
    if LHplot:
        ax1.set_ylabel('Normalised Flux')
        ax2.set_ylabel('Normalised Flux')
        LHplot = False
    ax2.set_xlabel('Orbital Phase')
    ax2.yaxis.set_major_locator(MaxNLocator(4,prune='both'))		

for ax in plt.gcf().get_axes()[::2]:
    ax.set_ylim([minY,maxY])
    ax.yaxis.set_major_locator(MaxNLocator(prune='both'))

plt.show()
