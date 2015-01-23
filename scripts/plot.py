from matplotlib import pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import seaborn
seaborn.set_style("white")

def readFile(file):
    x,y,e,fit,am,m,l,u = np.loadtxt(file).T
    return x,y,e,fit,am,m,l,u
    

x,y,e,fit,am,m,l,u = readFile('wasp12_rfit.dat')

x -= np.floor(x.min())

# remove airmass term from best fit and data
y /= am
fit /= am



gs = gridspec.GridSpec(2,1,height_ratios=[2,1])
gs.update(hspace=0.0)

ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[1,0],sharex=ax1)

ax1.errorbar(x,y-m,yerr=e,fmt='.')
ax1.plot(x,fit,'k-',lw=2)
ax2.errorbar(x,y-fit,yerr=e,fmt='.',alpha=0.5)
ax2.plot(x,m,'k-')
ax2.fill_between(x,l,u,color='k',alpha=0.7)

plt.setp(ax1.get_xticklabels(),visible=False)
ax2.yaxis.set_major_locator(MaxNLocator(4,prune='both'))
ax1.yaxis.set_major_locator(MaxNLocator(9,prune='lower'))

ax1.set_ylabel('Brightness')
ax2.set_ylabel('Residuals')
ax2.set_xlabel('Time (days)')
plt.show()