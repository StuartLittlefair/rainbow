import math as m
import numpy as np

def mean2ecc(manom, e, acc=1.e-10):
    """
    Solves Kepler's equation E - e sin(E) = M to derive the eccentric anomaly E
    from the mean anomaly M

    manom  -- mean anomoly in radians (=2.*pi times usual orbital phase measured
              from periastron). Can be a numpy array.
    e      -- orbital eccentricity < 1
    acc    -- accuracy in radians for the solution. 

    Returns the eccentric anomalies in radians
    """

    # First map mean anomaly into range 0 to 2*pi
    man = manom - 2.*np.pi*np.floor(manom/(2.*np.pi))
    
    # Now solve for eccentric anomaly using Newton-Raphson applied to Kepler's equation
    ecc = man
    old = man - man
    while np.abs(ecc-old).max() > acc: 
        old = ecc
        ecc = ecc - (ecc - e*np.sin(ecc) - man)/(1.-e*np.cos(ecc))
    return ecc

def true2ecc(true, e):
    """
    Returns the eccentric anomaly E given the true anomaly T using the relation

    tan(E/2) = [sqrt(1-e)/sqrt(1+e)] * tan(T/2)

    ecc   -- eccentric anomaly, can be an array
    e     -- the eccentricity

    Returns the eccentric anomalies in radians
    """
    return 2.*np.arctan2(np.sqrt(1.-e)*np.sin(true/2.), np.sqrt(1.+e)*np.cos(true/2.)) 

def ecc2true(ecc, e):
    """
    Returns the true anomaly T given the eccentric anomaly E using the relation

    tan(T/2) = [sqrt(1+e)/sqrt(1-e)] * tan(E/2)

    ecc   -- eccentric anomaly, can be an array
    e     -- the eccentricity

    Returns the true anomalies in radians
    """
    return 2.*np.arctan2(np.sqrt(1.+e)*np.sin(ecc/2.), np.sqrt(1.-e)*np.cos(ecc/2.)) 

def ecc2mean(ecc, e):
    """
    Returns the mean anomaly M from the eccentric anomaly E

    ecc   -- eccentric anomaly, can be an array
    e     -- the eccentricity

    Returns the mean anomalies in radian
    """
    return ecc - e*np.sin(ecc)
    
