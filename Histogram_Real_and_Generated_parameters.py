import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import math 
from numpy.random import randn




#### Simulation(real) data reading from file ########

columns=['Energy','Length', 'Width', 'Dist', 'Alpha', 'F2', 'Size','Asym']
data=pd.read_csv("HillasGamma_Zen05_PE18.dat",sep='\s+',usecols=columns)

data=data[(data.Alpha<89.0) & (data.Alpha>0.0)]
data.Alpha=-np.log(3.14*data.Alpha/180.0) 
data.Energy=np.log10(data.Energy)
data.Size=np.log10(data.Size)
# Normalize the input data(i.e. mean=0,std=1.0) 
data=data[data.Asym>0.0]   ### assending order of size 
#data = data.('Asym', ascending=True)
data=(data-data.mean(axis=0))/np.std(data)


##### Generate latent random points ######
def generate_latent_points(latent_dim, n):
    # generate points in the latent space
    x_input = randn(latent_dim * n)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n, latent_dim)
    return x_input
#lat_dim=20 ; #latent dimension 
DL=len(data)                  #len(data)

### 10K latent points(hence, 10K synthetic events are being generated)
x_input = generate_latent_points(latent_dim, 10000)



####################Output from trained generator########################
X=generator.predict(x_input) # Predicting using a trained genarator
##### Noirmalization of generated data##############3
X=(X-np.mean(X))/np.std(X)



####################### HILLAS parameter plot starts here  #################################

NB=20 # Nos of bins
R=3 # Nos of Rows
C=3 # Nos of columns
fig,ax=plt.subplots(R,C)
#fig.suptitle("Real Vs Synthetic(GAN) data", fontsize=15 ,  fontweight='bold' ,color='blue')


############# ENERGY plot #################################
R1=0
C1=0

(aE,bE,cE)=ax[R1,C1].hist(X[:,0],bins=NB,histtype='step',density=True,color='white',label='GAN',lw=1.5,range=(-2.14,3.5))

ax[R1,C1].hist(bE[:-1], bE, weights=aE , histtype='step',density=True,color='blue',label='GAN',lw=1.5)
ax[R1,C1].hist(data.iloc[:,0],bins=NB,histtype='step',density=True,color='red',label='Real',lw=1.5 ,range=(-2.14,3.5),ls='--')
ax[R1,C1].set_xlabel(r'$\mathbf{log_{10}(Energy)}$')

################## LENGTH PLOT #################################
R1=0
CN=1 ### Column no for data , 0-Length,1- Width, 2- log(Size)

(aL,bL,cL)=ax[R1,CN].hist(X[:,CN],bins=NB,histtype='step',density=True,color='white',label='GAN',lw=1.5 , range=(-4.0,4.0))

ax[R1,CN].hist(bL[:-1], bL, weights=aL , histtype='step',density=True,color='blue',label='GAN',lw=1.5)
ax[R1,CN].hist(data.iloc[:,CN],bins=NB,histtype='step',density=True,color='red',label='Real',lw=1.5,range=(-4.0,4.0),ls='--')
ax[R1,CN].set_xlabel(r'$\mathbf{Length}$')

#ax[CN,CN].set_ylabel(r'$\mathbf{Norm. Freq.}$')
#ax[CN,CN].legend()


################### WIDTH PLOT##############################
C1=2 ### Column no for data , 0-Length,1- Width, 2- log(Size)

(aW,bW,cW)=ax[R1,C1].hist(X[:,2],bins=NB,histtype='step',density=True,color='white',label='GAN',range=(-2.5,3.2) , lw=1.5)

ax[R1,C1].hist(bW[:-1], bW, weights=aW , histtype='step',density=True,color='blue',label='GAN',lw=1.5)
(a,b,c)=ax[R1,C1].hist(data.Width,bins=NB,histtype='step',density=True,color='red',label='Real',range=(-2.5,3.2), lw=1.5,ls='--')
ax[R1,C1].set_xlabel('$\mathbf{Width}$')
#ax[R1,C1].set_ylabel(r'$\mathbf{Norm. Freq.}$')

#ax[R1,C1].legend()

############### DISTANCE PLOT ################################################################
C1=0 ### Column no for data , 0-Length,1- Width, 2- log(Size)
R2=1 ## Column No

(aD,bD,cD)=ax[R2,C1].hist(X[:,3],bins=NB,histtype='step',density=True,color='white',label='GAN' , range=(-2.6,2.63),lw=1.5)


binCenter=0.5*(bD[1:]+bD[:-1])
GAN_err=(aD*(b[1]-b[0])/len(X))**0.5


ax[R2,C1].hist(bD[:-1], bD, weights=aD , histtype='step',density=True,color='blue',label='GAN',lw=1.5 , range=(-2.6,2.63))
#ax[R2,C1].errorbar(binCenter,aD,yerr=GAN_err,ecolor='green',ls='None')
ax[R2,C1].hist(data.Dist,bins=NB,histtype='step',density=True,color='red',label='Real', range=(-2.6,2.6),lw=1.5,ls='--')
ax[R2,C1].set_xlabel(r'$\mathbf{Distance}$')
ax[R2,C1].set_ylim(0.0,0.6)


############# ALPHA PLOT  #############################
R2=1
C1=1 ### Column no for data , 0-Length,1- Width, 2- log(Size)
(a2,b2,c2)=ax[R2,C1].hist(X[:,7],bins=NB,histtype='step',density=True,color='white',label='GAN',lw=1.5 , range=(-0.6,4.0))


ax[R2,C1].hist(b2[:-1], b2, weights=a2 , histtype='step',density=True,color='blue',label='GAN',lw=1.5 , range=(-0.6,4.0))

ax[R2,C1].hist(data.Alpha,bins=NB,histtype='step',density=True,color='red',label='Real',lw=1.5, range=(-0.6,4.0),ls='--')
ax[R2,C1].set_xlabel(r'$\mathbf{log(Alpha)}$')
ax[R2,C1].set_ylim(0.0,)
#ax[R2,C1].legend()


####################### Frac2 PLOT ##########################
C1=2 ### Column no for data , 0-Length,1- Width, 2- log(Size)
(aF,bF,cF)=ax[R2,C1].hist(X[:,4],bins=NB,histtype='step',density=True,color='white',label='GAN' , range=(-2.7,2.63),lw=1.5)

ax[R2,C1].hist(bF[:-1], bF, weights=aF , histtype='step',density=True,color='blue',label='GAN',lw=1.5 , range=(-2.7,2.63))
ax[R2,C1].hist(data.F2,bins=NB,histtype='step',density=True,color='red',label='Real',range=(-2.7,2.63),lw=1.5,ls='--')
ax[R2,C1].set_xlabel(r'$\mathbf{Frac2}$')
ax[R2,C1].set_ylim(0.0,0.45)
#ax[R2,C1].legend()


###############log(SIZE) PLOT #############################
R3=2
C1=0 ### Column no for data , 0-Length,1- Width, 2- log(Size)
ax[R3,C1].hist(X[:,6],bins=NB,histtype='step',density=True,color='blue',label='GAN',lw=1.5 , range=(-2.04,4.0))
ax[R3,C1].hist(data.Size,bins=NB,histtype='step',density=True,color='red',label='Real',lw=1.5 , range=(-2.04,4.0) , ls='--')
ax[R3,C1].set_xlabel(r'$\mathbf{log_{10}(Size)}$')
ax[R3,C1].set_ylim(0.0,0.6)


################## Asymmetry PLOT ########################
C1=1
(aAS,bAS,cAS)=ax[R3,C1].hist(X[:,5],bins=NB,histtype='step',density=True,color='white',label='GAN',lw=1.5 , range=(-2.14,2.0))
ax[R3,C1].hist(bAS[:-1], bAS, weights=aAS , histtype='step',density=True,color='blue',label='GAN',lw=1.5 , range=(-2.14,2.0))

ax[R3,C1].hist(data.Asym,bins=NB,histtype='step',density=True,color='red',label='Real',lw=1.5 ,range=(-2.14,2.0), ls='--')
ax[R3,C1].set_xlabel(r'$\mathbf{Asym}$')
ax[R3,C1].set_ylim(0.0,1.5)

####### MAKING PLOT OFF(INVISIBLE) #######################

C1=2
ax[R3,C1].axis('off')
############################ HILLAS parameters plot ends here ############################################################################






