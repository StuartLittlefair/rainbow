'''
classes and helper functions to handle simultaneously fitting multi-coloured
observations of exoplanet transits, with red noise being treated as a GP'''
from transit import transit
from utils.mcmc_utils import Param, Prior
import numpy as np
from collections import MutableSequence

# function to fit (quadratic limb darkening law)
fitFunc = transit.occultquad

try:
    import george
    def chooseKernel(name):
        if name == "ExpKernel":
            return george.kernels.ExpKernel
        if name == "ExpSquaredKernel":
            return george.kernels.ExpSquaredKernel        
        if name == "Matern32Kernel":
            return george.kernels.Matern32Kernel        
        if name == "Matern52Kernel":
            return george.kernels.Matern52Kernel        
        raise Exception('Kernel name not valid')
    
    class TransitModelGP(MutableSequence):
            '''transit model for  multiple bands.
            Can be passed to routines for calculating model and chisq, and prior prob
        
            can add bands at will with routine addBand. All bands share t0, b and rs_a.
            for each band only one limb darkening parameter varies, all others held constant
        
            Also behaves like a list of the current values of all params which vary - 
            this allows it to be seamlessly used with emcee'''
                
            # arguments are Param objects (see mcmc_utils)
            def __init__(self,period,t0,b,rs_a,kernel,tau,rp_rs,f0,u1,u2,A,B,rn_amp):
                    self.period = period
                    self.t0 = t0
                    self.b  = b
                    self.rs_a = rs_a
                    self.kernel = chooseKernel(kernel)
                    self.tau = tau
                    self.rp_rs = [rp_rs]
                    self.f0 = [f0]
                    self.u1 = [u2]
                    self.u2 = [u2]
                    self.A  = [A]
                    self.B  = [B]
                    self.rn_amp = [rn_amp]
                    self.ncolours = 1
                    # initialise list with all variable parameters
                    self.data = [self.t0, \
                            self.b, \
                            self.rs_a, \
                            self.tau, \
                            self.rp_rs[0], \
                            self.f0[0], \
                            self.u2[0], \
                            self.A[0], \
                            self.B[0], \
                            self.rn_amp[0] ]
                        
            def addBand(self,rp_rs,f0,u1,u2,A,B,rn_amp):
                    self.rp_rs.append(rp_rs)
                    self.f0.append(f0)
                    self.u1.append(u1)
                    self.u2.append(u2)
                    self.A.append(A)
                    self.B.append(B)
                    self.rn_amp.append(rn_amp)
                    self.ncolours += 1
                    # add parameters which vary to list
                    self.extend([self.rp_rs[-1], \
                            self.f0[-1], \
                            self.u2[-1], \
                            self.A[-1], \
                            self.B[-1], \
                            self.rn_amp[-1] ])
        
            def calc(self,band,x):
                # pars are: 
                # [t0, b, rs_a, rp_rs, f0, u2, A, B]
                #
                # we also have u1 for the limb darkening law, which is fixed
                # split into params for transit model (in each band): 
                # [t0, b, rs_a, rp_rs, f0, u1, u2] 
                # and params for 2nd order polynomial
                # [A,B]
                transitPars = [ \
                    self.t0.currVal, \
                    self.b.currVal, \
                    self.rs_a.currVal, \
                    self.rp_rs[band].currVal, \
                    self.f0[band].currVal, \
                    self.u1[band].currVal, \
                    self.u2[band].currVal]
                a,b = ( self.A[band].currVal, self.B[band].currVal )
                transitShape = transit.modeltransit(transitPars,fitFunc,self.period,x)

                # t is x rescaled so that it runs from -1 to 1
                t = -1.0 + 2.0*(x - x.min())/(x.max()-x.min())
                airmassFunc =  1.0 + a*t + b*t*t
                return transitShape*airmassFunc
    
            def chisq(self,x,y,yerr):
                retVal = 0.0
                for icol in range(self.ncolours):
                    resids = ( y[icol] - self.calc(icol,x[icol]) ) / yerr[icol]
                    retVal += np.sum(resids*resids) 
                return retVal
        
            def reducedChisq(self,x,y,yerr):
                return self.chisq(x,y,yerr) / (np.size(x) - self.npars - 1)

            def ln_prior(self):
                retVal = 0.0
                prior_pars_shared = ['t0','b','rs_a','tau']
                prior_pars_unique = ['rp_rs','f0','u1','u2','A','B','rn_amp']
                ncol = self.ncolours
                for par in prior_pars_shared:
                    param = getattr(self,par)
                    retVal += param.prior.ln_prob(param.currVal)
                for par in prior_pars_unique:
                    parArr = getattr(self,par)         
                    for icol in range(ncol):
                        param = parArr[icol]
                        retVal += param.prior.ln_prob(param.currVal)
                return retVal
       
            def ln_likelihood(self,x,y,yerr):
                ln_like = 0.0
                for icol in range(self.ncolours):
                    gp = george.GP(self.rn_amp[icol].currVal * self.kernel(self.tau.currVal))
                    gp.compute(x[icol],yerr[icol])
                    ln_like += gp.lnlikelihood(y[icol]-self.calc(icol,x[icol]))
                return ln_like
                    
            def lnprob(self,x,y,e):
                lnp = self.ln_prior()
                if np.isfinite(lnp):
                    return lnp + self.ln_likelihood(x,y,e)
                else:
                    return lnp

            def sample_conditional(self,band,x,y,yerr):
                gp = george.GP(self.rn_amp[band].currVal * self.kernel(self.tau.currVal))
                gp.compute(x,yerr)
                res = y - self.calc(band,x)
                samples = gp.sample_conditional(res, x, size=300)
                return samples
            
            def calc_airmass_term(self,band,x):
                a,b = ( self.A[band].currVal, self.B[band].currVal )
                # t is x rescaled so that it runs from -1 to 1
                t = -1.0 + 2.0*(x - x.min())/(x.max()-x.min())
                airmassFunc =  1.0 + a*t + b*t*t
                return airmassFunc 

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
                    
except ImportError, e:
    pass # module doesn't exit, no nice GP stuff for you
    
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
    
        def calc(self,band,x):
            # pars are: 
            # [t0, b, rs_a, rp_rs, f0, u2, A, B]
            #
            # we also have u1 for the limb darkening law, which is fixed
            # split into params for transit model (in each band): 
            # [t0, b, rs_a, rp_rs, f0, u1, u2] 
            # and params for 2nd order polynomial
            # [A,B]
            transitPars = [ \
                self.t0.currVal, \
                self.b.currVal, \
                self.rs_a.currVal, \
                self.rp_rs[band].currVal, \
                self.f0[band].currVal, \
                self.u1[band].currVal, \
                self.u2[band].currVal]
            a,b = ( self.A[band].currVal, self.B[band].currVal )
            transitShape = transit.modeltransit(transitPars,fitFunc,self.period,x)

            # t is x rescaled so that it runs from -1 to 1
            t = -1.0 + 2.0*(x - x.min())/(x.max()-x.min())
            airmassFunc =  1.0 + a*t + b*t*t
            return transitShape*airmassFunc

        def chisq(self,x,y,yerr):
            retVal = 0.0
            for icol in range(self.ncolours):
                resids = ( y[icol] - self.calc(icol,x[icol]) ) / yerr[icol]
                retVal += np.sum(resids*resids) 
            return retVal
    
        def reducedChisq(self,x,y,yerr):
            return self.chisq(x,y,yerr) / (np.size(x) - self.npars - 1)

        def ln_prior(self):
            retVal = 0.0
            prior_pars_shared = ['t0','b','rs_a']
            prior_pars_unique = ['rp_rs','f0','u1','u2','A','B']
            ncol = self.ncolours
            for par in prior_pars_shared:
                param = getattr(self,par)
                retVal += param.prior.ln_prob(param.currVal)
            for par in prior_pars_unique:
                parArr = getattr(self,par)         
                for icol in range(ncol):
                    param = parArr[icol]
                    retVal += param.prior.ln_prob(param.currVal)
            return retVal

        def ln_likelihood(self,x,y,yerr):
            errFac = 0.0
            for icol in range(self.ncolours):
                errFac += np.sum( np.log (2.0*np.pi*yerr[icol]**2) )
            return -0.5*(errFac + self.chisq(x,y,yerr))
                
        def lnprob(self,x,y,e):
            lnp = self.ln_prior()
            if np.isfinite(lnp):
                return lnp + self.ln_likelihood(x,y,e)
            else:
                return lnp
        
        def calc_airmass_term(self,band,x):
            a,b = ( self.A[band].currVal, self.B[band].currVal )
            # t is x rescaled so that it runs from -1 to 1
            t = -1.0 + 2.0*(x - x.min())/(x.max()-x.min())
            airmassFunc =  1.0 + a*t + b*t*t
            return airmassFunc 

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
                

      


                     


        

