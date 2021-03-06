import numpy as np
from scipy.interpolate import interp2d, SmoothBivariateSpline
import pkg_resources

def make_ld_funcs(data,band,teff):
    assert band in ['u','g','r','i','z']
    band += '*'
    if teff < 3500:
        atmos = 'P'
    else:
        atmos = 'A'
    mask = (data['Filt'] == band) & (data['Mod'] == atmos) & (data['Met'] == 'L')

    x=data[mask]['logg'] #logg
    y=data[mask]['Teff'] #teff
    z1=data[mask]['a'] #quad ld coefficient, a
    z2=data[mask]['b'] #quad ld coefficient, b
    func1 = SmoothBivariateSpline(x,y,z1)
    func2 = SmoothBivariateSpline(x,y,z2)
    return (func1,func2)

def ld (funcs,logg,teff):
    func1, func2 = funcs
    a = func1(logg,teff)[0]
    b = func2(logg,teff)[0]
    return a, b

def main(): 
    import warnings
    from astropy import table
    ldFile = pkg_resources.resource_filename('rainbow','data_files/claret-bloemen.vot')
    
    logg, gerr = raw_input('> Give log g and error: ').split()
    teff, terr = raw_input('> Give eff. temp. and error: ').split()
    logg = float(logg); gerr = float(gerr)
    teff = float(teff); terr = float(terr)

    print 'reading data table'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = table.Table.read(ldFile)
    print 'calculating limb darkening'    
    gvals=np.random.normal(loc=logg,scale=gerr,size=100)
    tvals=np.random.normal(loc=teff,scale=terr,size=100)

    for band in ['u', 'g', 'r', 'i', 'z']:
        funcs = make_ld_funcs(data,band,teff)
        a = [] 
        b= []
        for g,t in zip(gvals,tvals):
            this_a, this_b = ld(funcs,g,t)
            a.extend(this_a)
            b.extend(this_b)
        print '%s band LD coeff (a) = %f +/- %f' % (band, np.median(a),np.std(a))
        print '%s band LD coeff (b) = %f +/- %f' % (band, np.median(b),np.std(b))


if __name__ == "__main__":
    main()
