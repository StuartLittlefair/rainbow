import transit
from utils.mcmc_utils import Param, Prior
import numpy as np
from collections import MutableSequence

# function to fit (quadratic limb darkening law)
fitFunc = transit.occultquad

class TransitModel(MutableSequence):
        '''transit model for  multiple bands.
        Can be passed to routines for calculating model and chisq, and prior prob
        
        can add bands at will with routine addBand. All bands share t0, b and rs_a.
        for each band only one limb darkening parameter varies, all others held constant
        
        Also behaves like a list of the current values of all params which vary - 
        this allows it to be seamlessly used with emcee'''
                
        # arguments are Param objects (see mcmc_utils)
        def __init__(self,period,t0,b,rs_a,rp_rs,f0,u1,u2,A,B):
                self.period = period
                self.t0 = t0
                self.b  = b
                self.rs_a = rs_a
                self.rp_rs = [rp_rs]
                self.f0 = [f0]
                self.u1 = [u2]
                self.u2 = [u2]
                self.A  = [A]
                self.B  = [B]
                self.ncolours = 1
                # initialise list with all variable parameters
                self.data = [self.t0, \
                        self.b, \
                        self.rs_a, \
                        self.rp_rs[0], \
                        self.f0[0], \
                        self.u2[0], \
                        self.A[0], \
                        self.B[0] ]
                        
        def addBand(self,rp_rs,f0,u1,u2,A,B):
                self.rp_rs.append(rp_rs)
                self.f0.append(f0)
                self.u1.append(u1)
                self.u2.append(u2)
                self.A.append(A)
                self.B.append(B)
                self.ncolours += 1
                # add parameters which vary to list
                self.extend([self.rp_rs[-1], \
                        self.f0[-1], \
                        self.u2[-1], \
                        self.A[-1], \
                        self.B[-1] ])
                        
        def __getitem__(self,ind):
                return self.data[ind].currVal
        def __setitem__(self,ind,val):
                self.data[ind].currVal = val
        def __delitem__(self,ind):
                self.data.remove(ind)
        def __len__(self):
                return len(self.data)
        def insert(self,ind,val):
                self.data.insert(ind,val)
        @property
        def npars(self):
                return len(self.data)
                        
def parseInput(file):
        blob = np.loadtxt(file,dtype='string',delimiter='\n')
        input_dict = {}
        for line in blob:
                k,v = line.split('=')
                input_dict[k.strip()] = v.strip()
        return input_dict

def parseParam(parString):
        fields = parString.split()
        val = float(fields[0])
        priorType = fields[1].strip()
        priorP1   = float(fields[2])
        priorP2   = float(fields[3])
        return Param(val, Prior(priorType, priorP1, priorP2))
                
def calc_model(thisModel,band,x):
        # pars are: 
        # [t0, b, rs_a, rp_rs, f0, u2, A, B]
        #
        # we also have u1 for the limb darkening law, which is fixed
        # split into params for transit model (in each band): 
        # [t0, b, rs_a, rp_rs, f0, u1, u2] 
        # and params for 2nd order polynomial
        # [A,B]
        transitPars = [ \
                thisModel.t0.currVal, \
                thisModel.b.currVal, \
                thisModel.rs_a.currVal, \
                thisModel.rp_rs[band].currVal, \
                thisModel.f0[band].currVal, \
                thisModel.u1[band].currVal, \
                thisModel.u2[band].currVal]
        #print transitPars
        a,b = ( thisModel.A[band].currVal, thisModel.B[band].currVal )
        transitShape = transit.modeltransit(transitPars,fitFunc,thisModel.period,x)
        
        # t is x rescaled so that it runs from -1 to 1
        t = -1.0 + 2.0*(x - x.min())/(x.max()-x.min())
        airmassFunc =  1.0 + a*t + b*t*t
        return transitShape*airmassFunc

def calc_airmass_model(thisModel,band,x):
        # pars are: 
        # [t0, b, rs_a, rp_rs, f0, u2, A, B]
        #
        # we also have u1 for the limb darkening law, which is fixed
        # split into params for transit model (in each band): 
        # [t0, b, rs_a, rp_rs, f0, u1, u2] 
        # and params for 2nd order polynomial
        # [A,B]
        transitPars = [ \
                thisModel.t0.currVal, \
                thisModel.b.currVal, \
                thisModel.rs_a.currVal, \
                thisModel.rp_rs[band].currVal, \
                thisModel.f0[band].currVal, \
                thisModel.u1[band].currVal, \
                thisModel.u2[band].currVal]
        #print transitPars
        a,b = ( thisModel.A[band].currVal, thisModel.B[band].currVal )

        # t is x rescaled so that it runs from -1 to 1
        t = -1.0 + 2.0*(x - x.min())/(x.max()-x.min())
        airmassFunc =  1.0 + a*t + b*t*t
        return airmassFunc
        
def chisq(thisModel,x,y,yerr):
        retVal = 0.0
        for icol in range(thisModel.ncolours):
                resids = ( y[icol] - calc_model(thisModel,icol,x[icol]) ) / yerr[icol]
                retVal += np.sum(resids*resids) 
        return retVal
        
def reducedChisq(thisModel,x,y,yerr):
        return chisq(thisModel,x,y,yerr) / (np.size(x) - thisModel.npars - 1)

def ln_prior(thisModel):
        retVal = 0.0
        prior_pars_shared = ['t0','b','rs_a']
        prior_pars_unique = ['rp_rs','f0','u1','u2','A','B']
        ncol = thisModel.ncolours
        for par in prior_pars_shared:
                param = getattr(thisModel,par)
                retVal += param.prior.ln_prob(param.currVal)
        for par in prior_pars_unique:
                parArr = getattr(thisModel,par)         
                for icol in range(ncol):
                        param = parArr[icol]
                        retVal += param.prior.ln_prob(param.currVal)
        return retVal

def ln_likelihood(thisModel,x,y,yerr):
        errFac = 0.0
        for icol in range(thisModel.ncolours):
                errFac += np.sum( np.log (2.0*np.pi*yerr[icol]**2) )
        return -0.5*(errFac + chisq(thisModel,x,y,yerr))
                
def lnprob(pars,thisModel,x,y,e):
        # ln_likelihood = -chisq/2
        # ln_posterior = ln_likelihood + ln_prior
        
        # we need to update the model we're using to use pars as submitted by MCMC
        for i in range(thisModel.npars):
                thisModel[i]=pars[i]
        lnp = ln_prior(thisModel)
        if np.isfinite(lnp):
                return lnp + ln_likelihood(thisModel,x,y,e)
        else:
                return lnp

