import numpy as np 
import matplotlib.pyplot as plt 

def s(l,n,b,alpha):
	return(l+n+b*(1-alpha))


#BK -jet (except b)
#b = np.linspace(0,-2,1000)
#l = np.linspace(0.5,1,5)
b_parabolic = np.linspace(0*0.5,-2*0.5,1000)
b_conical = np.linspace(0*1,-2*1,1000)

plt.plot(b_parabolic,s(0.5,-2,b_parabolic,-0.5),marker='+',linewidth=0.5,markersize=0.5,label='l=0.5',color='red')
plt.plot(b_conical,s(1,-2,b_conical,-0.5),marker='+',linewidth=0.5,markersize=0.5,label='l= 1',color='blue')
plt.vlines(-0.5,-4,-1,linestyle = '--',label='torodial B,parabolic',color='grey')
plt.vlines(-1,-4,-1,linestyle = '--',label='polodial B,parabolic',color='black')

plt.vlines(-1,-4,-1,linestyle = 'dotted',label='torodial B, conical',color='grey')
plt.vlines(-2,-4,-1,linestyle = 'dotted',label='polodial B, conical',color='black')

plt.xlabel('b')
plt.ylabel('s')
plt.legend()
plt.savefig('./b-parameter_space.pdf',dpi=300)
plt.close()