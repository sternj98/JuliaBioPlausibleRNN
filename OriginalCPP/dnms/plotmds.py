
from pylab import *
import numpy as np
import scipy as sp
import glob


from sklearn import manifold
#from sklearn.metrics import euclidean_distances

if 1:
    datalist=[]
    labellist=[]
    for trialtype in range(4):
        print(trialtype)

#trying 300 ms , total time 1100
#0 : too messy
# 1 : also too messy
# Also too messy.

# Back to 1000 ms, eval time 200, but longer training?
# 0 still seems least bad
        fnames = glob.glob('rs_long_type'+str(trialtype)+'*ETA0.1*SEED0.txt')
        for nm in fnames:
            r = loadtxt(nm)
            #z = r.reshape((110,10,200))
            #z = sum(z,axis=1)
            z = r #r[0::10,:]
            datalist.append(z)
            #labellist.append([trialtype]*r.shape[0])
            labellist.append(trialtype)



matdata = dstack(datalist) ; #+ .5  * standard_normal(matdata.shape)

matdata = matdata[:,:,::2]
NBPTS = matdata.shape[2]

matdata += .0 * standard_normal(shape(matdata))


ion()

fgr, sps = subplots(3, 3)

# 8.5cm = 3.3 inches for single column. 6.9 inches for two-column
#fgr.set_size_inches(3.3, 6)
#fgr.set_facecolor('white')
#slicetimes= [850, 900, 990, 999] #[200, 600, 900 , 850, 990, 999]
slicetimes= [199, 599, 799, 999] #[200, 600, 900 , 850, 990, 999]

subplots_adjust(wspace=0, hspace=.25)

for numgraph in range(4):

    tslc = matdata[slicetimes[numgraph],:,:].T

    mds = manifold.MDS(n_components=2,  max_iter=1000, dissimilarity="euclidean")
    pos = mds.fit(tslc).embedding_

    ax = sps[round(numgraph/2),numgraph%2]

    ax.set_title(str(1+slicetimes[numgraph])+'ms', size=10)

    ax.plot(pos[0:round(NBPTS/4)-1, 0], pos[0:round(NBPTS/4)-1, 1], 'oc', markersize=8)
    ax.plot(pos[round(NBPTS/4):round(2*NBPTS/4-1), 0], pos[round(NBPTS/4):round(2*NBPTS/4-1), 1], 'or', markersize=8)
    ax.plot(pos[round(2*NBPTS/4):round(3*NBPTS/4-1), 0], pos[round(2*NBPTS/4):round(3*NBPTS/4-1), 1], 'og', markersize=8)
    ax.plot(pos[round(3*NBPTS/4):NBPTS-1, 0], pos[round(3*NBPTS/4):NBPTS-1, 1], 'oy', markersize=8)
    if numgraph==0 or numgraph==2:
        ax.set_ylabel('Dimension 2', size=10)
    if numgraph==2 or numgraph==3:
        ax.set_xlabel('Dimension 1', size=10)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(-10,10)
    ax.set_ylim(-10,10)
    ax.set_aspect('equal', 'box')

sps[2,0].axis('off')
sps[2,1].axis('off')
sps[0,2].axis('off')
sps[1,2].axis('off')
sps[2,2].axis('off')
sps[0,1].legend(['AA','AB', 'BA', 'BB'],  numpoints = 1, ncol= 1, loc=3, prop={'size':10}, bbox_to_anchor=(1.1,-.5))
savefig('figure_mds.png', bbox_inches='tight', dpi=300)

draw()
