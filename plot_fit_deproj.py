import numpy as np 
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.stats import linregress as linreg
from scipy.stats import chisquare as chisq
import os
import shutil
from scipy.optimize import curve_fit as cf
from scipy.stats import skew,kurtosis
import tqdm 
from collections import OrderedDict
from matplotlib.patches import Rectangle
from scipy.optimize import least_squares as ls
from scipy.stats import chisquare
import matplotlib.cm as cm
import matplotlib

Name = np.load('./numpy_arrays/list_info/names.npy')
c_name = np.load('./numpy_arrays/list_info/c_names.npy')
classes = np.load('./numpy_arrays/list_info/classes.npy')

alpha = np.load('./numpy_arrays/list_info/alphas.npy')
pcpmas = np.load('./numpy_arrays/list_info/pcpmas.npy')
theta_min = np.load('./numpy_arrays/list_info/theta_min.npy')
theta_max = np.load('./numpy_arrays/list_info/theta_max.npy')
alpha_err = 0.15 * np.abs(alpha)


pwd = os.getcwd()

def B(r,b,B_x,c_dist):
	return(B_x/((c_dist)**(b)) * r**(b))

#give BH mass in Solar mass to get grav_rad in pc,au,m,cm
def grav_rad(M):
	rg = (const.G * M * 1.98847 * 10**(30))/((const.c)**2)#3.085*10**(16)*
	return([rg/(3.085*10**(16)),rg/149597870700,rg,rg*100]) 

def kerr(edot,M,B):
	return(np.sqrt(edot*10**(-45)*(M/10**9)**(-2)*(B/10**(4))**(-2)))

def spin_BZ(P,B,M):
	return(np.sqrt(5)*(P/10**(44))**(0.5)*(B/10**4)**(-1)*(M/10**8)**(-1))
def spin_hybr(P,B,M):
	return((1.05)**(-0.5)*(P/10**(44))**(0.5)*(B/10**4)**(-1)*(M/10**8)**(-1))

def B_spin_BZ(P,M,j):
	return(np.sqrt(5)/j *(P/10**(44))**0.5 * (M/10**8)**(-1))

def B_kerr(edot,M,kerr):
	return(np.sqrt(edot*10**(-45)*(kerr)**(-2)*(M/10**9)**(-2))*10**4)
def B_eddington(M):
	return(10**4 * 6*np.sqrt(M/10**8))

def b_equip(s,l,alpha):
	return((s-l)/(3-alpha))
def b_pcons(s,l,alpha):
	return((s+l)/(1-alpha))

def db_equip(s,l,alpha,ds,dl,dalpha):
	return(((ds/(3-alpha))**2+(-dl/(3-alpha))**2+((s-l)/(3-alpha)**2)**2)**(0.5))
def db_pcons(s,l,alpha,ds,dl,dalpha):
	return(((ds/(1-alpha))**2+(-dl/(1-alpha))**2+((s+l)/(1-alpha)**2)**2)**(0.5))

def Dchisquared(obs,fit,ste):
	#var = np.var(obs)
	return(((obs-fit)/ste)**2)


def adjustFigAspect(fig,aspect=1):
	xsize,ysize = fig.get_size_inches()
	minsize = min(xsize,ysize)
	xlim = .35*minsize/xsize
	ylim = .35*minsize/ysize
	if aspect < 1:
		xlim *= aspect
	else:
		ylim /= aspect
	fig.subplots_adjust(left=.5-xlim,
						right=.5+xlim,
						bottom=.5-ylim,
						top=.5+ylim)

def forceAspect(ax,aspect=1):
	#aspect is width/height
	scale_str = ax.get_yaxis().get_scale()
	xmin,xmax = ax.get_xlim()
	ymin,ymax = ax.get_ylim()
	if scale_str=='linear':
		asp = abs((xmax-xmin)/(ymax-ymin))/aspect
	elif scale_str=='log':
		asp = abs((scipy.log(xmax)-scipy.log(xmin))/(scipy.log(ymax)-scipy.log(ymin)))/aspect
	ax.set_aspect(asp)




def get_data(freq,index):
	Name = np.load('./numpy_arrays/list_info/names.npy')
	data = np.load(pwd+'/data_test/'+str(Name[index])+'_'+str(freq)+'GHz.npy')
	return(data)


def plot_gradients(source_param,jet_param):

	d43 = get_data(43,source_param)
	d15 = get_data(15,source_param)
	fitfunc = lambda p, x: p[0] + p[1] * x   
	errfunc = lambda p, x, y, err: (y - fitfunc(p, x)) / err
	ppmas = pcpmas[source_param]
	theta = theta_max[source_param]

	if jet_param == 'l' and np.isnan(theta)==False:
		x43 = d43[0]*ppmas/np.sin(theta*(np.pi/180))
		y43 = d43[1]*ppmas
		x15 = d15[0]*ppmas/np.sin(theta*(np.pi/180))
		y15 = d15[1]*ppmas
		pinit = [1.0,1.0]
		mask15 = np.where(x15 <= np.max(x43))[0]
		mask43 = np.where(x43 >= np.min(x15))[0]
		err43 = d43[5]*ppmas
		err15 = d15[5]*ppmas
		jp = 'l'
		ylab = r'$d_j/\mathrm{pc}$'
		xlab = r'$d_c/\mathrm{pc}$ deprojected'
	elif jet_param == 'l' and np.isnan(theta)==True:
		x43 = d43[0]*ppmas
		y43 = d43[1]*ppmas
		x15 = d15[0]*ppmas
		y15 = d15[1]*ppmas
		pinit = [1.0,1.0]
		mask15 = np.where(x15 <= np.max(x43))[0]
		mask43 = np.where(x43 >= np.min(x15))[0]
		err43 = d43[5]*ppmas
		err15 = d15[5]*ppmas
		jp = 'l'
		ylab = r'$d_j/\mathrm{mas}$'
		xlab = r'$d_c/\mathrm{pc}$ projected'

	if jet_param == 's' and np.isnan(theta)==False:
		x43 = d43[0]*ppmas/np.sin(theta*(np.pi/180))
		y43 = d43[2] 
		x15 = d15[0]*ppmas/np.sin(theta*(np.pi/180))
		y15 = d15[2]  
		pinit = [1.0,-1.0]
		mask15 = np.where(x15 <= np.max(x43))[0]
		mask43 = np.where(x43 >= np.min(x15))[0]
		err43 = d43[4]
		err15 = d15[4]
		jp = 's'
		ylab = r'$t_b/\mathrm{K}$'
		xlab = r'$d_c/\mathrm{pc}$ deprojected'
	elif jet_param == 's' and np.isnan(theta)==True:
		x43 = d43[0]*ppmas
		y43 = d43[2] 
		x15 = d15[0]*ppmas
		y15 = d15[2]  
		pinit = [1.0,-1.0]
		mask15 = np.where(x15 <= np.max(x43))[0]
		mask43 = np.where(x43 >= np.min(x15))[0]
		err43 = d43[4]
		err15 = d15[4]
		jp = 's'
		ylab = r'$t_b/\mathrm{K}$'
		xlab = r'$_c/\mathrm{pc}$ projected'

	if jet_param == 'sd':
		x43 = d43[1]*ppmas
		y43 = d43[2]*ppmas
		x15 = d15[1]*ppmas
		y15 = d15[2]*ppmas
		pinit = [1.0,-1.0]
		mask15 = np.where(x15 <= np.max(x43))[0]
		mask43 = np.where(x43 >= np.min(x15))[0]
		err43 = d43[4]*ppmas
		err15 = d15[4]*ppmas
		jp=r'$s_d$'
		ylab = r'$t_b/\mathrm{K}$'
		xlab = r'$d_j/\mathrm{pc}$'
	

	epoch15 = d15[3]
	epoch43 = d43[3]


	#fit 15
	logx15 = np.log10(x15)
	logy15 = np.log10(y15)
	logyerr15 = err15 / y15
	out15 = ls(errfunc, pinit,args=(logx15, logy15, logyerr15),loss = 'cauchy',verbose=0)
	pfinal15 = out15.x
	index15 = pfinal15[1]
	amp15 = 10.0**pfinal15[0]
	x0_15 = np.linspace(np.min(x15),np.max(x15),x15.shape[0])
	residual15 = np.log10(y15)-np.log10(amp15*x15**index15)
	#rel_residual15 = (np.log10(y15)-np.log10(amp15*x15**index15))/(np.log10(y15)+np.log10(amp15*x15**index15))
	residual_stderror15 = np.sqrt(np.sum(residual15**2)/x15.shape[0])
	chi_temp15 = np.sum(Dchisquared(logy15,np.log10(amp15*x15**index15),logyerr15))
	chisq_red15 = chi_temp15/(x15.shape[0]-2)#(out15.nfev-3)
	error15 = np.sqrt(np.sum(residual15**2)/x15.shape[0])# * chisq_red15)
	error15_slope = error15
	fit15 = amp15*x0_15**index15
	amp15_p = 10.0**(pfinal15[0]+2*error15)
	amp15_m = 10.0**(pfinal15[0]-2*error15)
	schlauch15_p = amp15_p*x0_15**(index15)
	schlauch15_m = amp15_m*x0_15**(index15)
	fit15_p = amp15*x0_15**(index15+error15_slope)
	fit15_m = amp15*x0_15**(index15-error15_slope)

	#fit 15overlap
	logx15_ol = np.log10(x15[mask15])
	logy15_ol = np.log10(y15[mask15])
	logyerr15_ol = err15[mask15] / y15[mask15]
	out15_ol = ls(errfunc, pinit,args=(logx15_ol, logy15_ol, logyerr15_ol),loss = 'cauchy',verbose=0)
	pfinal15_ol = out15_ol.x
	index15_ol = pfinal15_ol[1]
	amp15_ol = 10.0**pfinal15_ol[0]
	x0_15_ol = np.linspace(np.min(x15[mask15]),np.max(x15[mask15]),x15[mask15].shape[0])
	residual15_ol = np.log10(y15[mask15])-np.log10(amp15_ol*x15[mask15]**index15_ol)
	residual_stderror15_ol = np.sqrt(np.sum(residual15_ol**2)/x15[mask15].shape[0])
	chi_temp15_ol = np.sum(Dchisquared(logy15_ol,np.log10(amp15_ol*x15[mask15]**index15_ol),logyerr15_ol))
	chisq_red15_ol = chi_temp15_ol/(x15[mask15].shape[0]-2)#(out15_ol.nfev-3)
	error15_ol = np.sqrt(np.sum(residual15_ol**2)/x15[mask15].shape[0])# * chisq_red15_ol)
	error15_slope_ol = error15_ol
	fit15_ol = amp15_ol*x0_15_ol**index15_ol
	fit15_ol_p = amp15_ol*x0_15_ol**(index15_ol+error15_ol)
	fit15_ol_m = amp15_ol*x0_15_ol**(index15_ol-error15_ol)

	#fit 43
	logx43 = np.log10(x43)
	logy43 = np.log10(y43)
	logyerr43 = err43 / y43
	out43 = ls(errfunc, pinit,args=(logx43, logy43, logyerr43),loss = 'cauchy',verbose=0)
	pfinal43 = out43.x
	index43 = pfinal43[1]
	amp43 = 10.0**pfinal43[0]
	x0_43 = np.linspace(np.min(x43),np.max(x43),x43.shape[0])
	residual43 = np.log10(y43)-np.log10(amp43*x43**index43)
	#rel_residual43 = (np.log10(y43)-np.log10(amp43*x43**index43))/(np.log10(y43)+np.log10(amp43*x43**index43))
	residual_stderror43 = np.sqrt(np.sum(residual43**2)/x43.shape[0])
	chi_temp43 = np.sum(Dchisquared(logy43,np.log10(amp43*x43**index43),logyerr43))
	chisq_red43 = chi_temp43 / (x43.shape[0]-2)#(out43.nfev-3)
	error43 = np.sqrt(np.sum(residual43**2)/x43.shape[0])# * chisq_red43)
	error43_slope = error43 
	fit43 = amp43*x0_43**index43
	fit43_p = amp43*x0_43**(index43+error43_slope)
	fit43_m = amp43*x0_43**(index43-error43_slope)
	amp43_p = 10.0**(pfinal43[0]+2*error43)
	amp43_m = 10.0**(pfinal43[0]-2*error43)
	schlauch43_p = amp43_p*x0_43**(index43)
	schlauch43_m = amp43_m*x0_43**(index43)
	
	#fit 43 overlap
	logx43_ol = np.log10(x43[mask43])
	logy43_ol = np.log10(y43[mask43])
	logyerr43_ol = err43[mask43] / y43[mask43]
	out43_ol = ls(errfunc, pinit,args=(logx43_ol, logy43_ol, logyerr43_ol),loss = 'cauchy',verbose=0)
	pfinal43_ol = out43_ol.x
	index43_ol = pfinal43_ol[1]
	amp43_ol = 10.0**pfinal43_ol[0]
	x0_43_ol = np.linspace(np.min(x43[mask43]),np.max(x43[mask43]),x43[mask43].shape[0])
	residual43_ol = np.log10(y43[mask43])-np.log10(amp43_ol*x43[mask43]**index43_ol)
	residual_stderror43_ol = np.sqrt(np.sum(residual43_ol**2)/x43[mask43].shape[0])
	chi_temp43_ol = np.sum(Dchisquared(logy43_ol,np.log10(amp43_ol*x43[mask43]**index43_ol),logyerr43_ol))
	chisq_red43_ol = chi_temp43_ol/ (x43[mask43].shape[0]-2) #(out43_ol.nfev-3)
	error43_ol = np.sqrt(np.sum(residual43_ol**2)/x43[mask43].shape[0])# * chisq_red43_ol
	error43_slope_ol = error43_ol
	fit43_ol = amp43_ol*x0_43_ol**index43_ol
	fit43_ol_p = amp43_ol*x0_43_ol**(index43_ol+error43_ol)
	fit43_ol_m = amp43_ol*x0_43_ol**(index43_ol-error43_ol)



	label0_43 = str(Name[source_param]) +', '+ str(43)+' GHz'+', '+'DOF: ' + str(x43.shape[0]) + ', ' + 'Epochs: ' + str(np.unique(d43[3]).shape[0]) + ', '
	label1_43 = jp +' ='+str(np.round(index43,2))+str(r'$\pm$')+str(np.round(error43_slope,2))+', '+r'$\chi ^2 $='+str(np.round(chi_temp43,3))+', '+r'$\chi ^2 _{red}$='+str(np.round(chisq_red43,3))
	label2_43 = jp +r'$_{overlap} =$'+str(np.round(index43_ol,2))+str(r'$\pm$')+str(np.round(error43_slope_ol,2))+', '+r'$\chi ^2 $='+str(np.round(chi_temp43_ol,3))+', '+r'$\chi ^2 _{red}$='+str(np.round(chisq_red43_ol,3))

	label0_15 = str(Name[source_param]) +', '+ str(15)+' GHz'+', '+'DOF: ' + str(x15.shape[0])+ ', '+ 'Epochs: ' + str(np.unique(d15[3]).shape[0]) + ', '
	label1_15 = jp +' ='+str(np.round(index15,2))+str(r'$\pm$')+str(np.round(error15_slope,2))+', '+r'$\chi ^2$='+str(np.round(chi_temp15,3))+', '+r'$\chi ^2 _{red}$='+str(np.round(chisq_red15,3))
	label2_15 = jp +r'$_{overlap} =$'+str(np.round(index15_ol,2))+str(r'$\pm$')+str(np.round(error15_slope_ol,2))+', '+r'$\chi ^2$='+str(np.round(chi_temp15_ol,3))+', '+r'$\chi ^2 _{red}$='+str(np.round(chisq_red15_ol,3))

	fig, axs = plt.subplots(2, 2,sharex=True)



	axs[0,0].axvspan(np.min(x43[mask43]), np.max(x43), alpha=0.5, color='grey',label='overlap region')
	## plot 43 l and fit
	
	#axs[0,0].errorbar(x43,y43,yerr=err*y43,linewidth=0,elinewidth=0.2,marker='+',c='B',markersize=0.5,label = label0_43)
	
	norm43 = matplotlib.colors.Normalize(vmin=min(epoch43), vmax=max(epoch43), clip=True)
	mapper43 = cm.ScalarMappable(norm=norm43, cmap='viridis')
	ep_color43 = np.array([(mapper43.to_rgba(v)) for v in epoch43])

	for x, y, e, color in zip(x43, y43, err43, ep_color43):
		#axs[0,0].plot(x, y, 'o', color=color)
		axs[0,0].errorbar(x, y, e, marker='+',elinewidth=0.2,markersize=0.5, color=color)
	

	axs[0,0].loglog(x0_43,fit43,linestyle='--',color='k',linewidth=0.5, label = label0_43 + label1_43)
	axs[0,0].loglog(x0_43_ol,fit43_ol,linestyle =':',color='grey',label = label2_43)
	axs[0,0].loglog(x0_43,fit43_p,linestyle='--',color='k',linewidth=0.5)
	axs[0,0].loglog(x0_43,fit43_m,linestyle='--',color='k',linewidth=0.5)
	#axs[0,0].loglog(x0_43,schlauch43_p,linestyle=':',color='k',linewidth=0.5)
	#axs[0,0].loglog(x0_43,schlauch43_m,linestyle=':',color='k',linewidth=0.5)
	axs[0,0].fill_between(x0_43,fit43_p,fit43_m,color = 'gainsboro')
	#axs[0,0].fill_between(x0_43,schlauch43_p,schlauch43_m,color = 'aquamarine')
	axs[0,0].set_xlim(10**(-1),10**3)
	axs[0,0].set_ylim(np.min(y43)*(1-0.1),np.max(y15)+1)
	axs[0,0].set_yscale('log')
	axs[0,0].set_xscale('log')
	axs[0,0].set_ylabel(ylab)
	axs[0,0].legend(loc=2,prop={'size': 3})
	## plot 43 dispersion
	axs[1,0].axvspan(np.min(x43[mask43]), np.max(x43), alpha=0.5, color='grey')

	for x, y, color in zip(x43, Dchisquared(logy43,np.log10(amp43*x43**index43),logyerr43), ep_color43):
		#axs[0,0].plot(x, y, 'o', color=color)
		#axs[1,0].errorbar(x, y, e, marker='+',elinewidth=0.2,markersize=0.5, color=color)
		axs[1,0].loglog(x,y,marker='+',linewidth=0,color = color,markersize=2)

	#axs[1,0].loglog(x43,Dchisquared(logy43,np.log10(amp43*x43**index43),logyerr43),marker='+',linewidth=0,color = 'B',markersize=2)
	#axs[1,0].set_ylim(-1,1)
	axs[1,0].set_xscale('log')
	axs[1,0].set_xlim(10**(-1),10**4)
	axs[1,0].set_ylabel(r'$\Delta \chi ^2$')

	##overlapregion  15
	axs[0,1].axvspan(np.min(x15), np.max(x15[mask15]), alpha=0.5, color='grey',label='overlap region')

	## plot 15 l and fit
	#axs[0,1].errorbar(x15,y15,yerr=err*y15,elinewidth=0.2,linewidth=0,marker='+',c='R',markersize=0.5,label = label0_15)
	
	
	norm15 = matplotlib.colors.Normalize(vmin=min(epoch15), vmax=max(epoch15), clip=True)
	mapper15 = cm.ScalarMappable(norm=norm43, cmap='plasma')
	ep_color15 = np.array([(mapper15.to_rgba(v)) for v in epoch15])

	for x, y, e, color in zip(x15, y15, err15, ep_color15):
		#axs[0,0].plot(x, y, 'o', color=color)
		axs[0,1].errorbar(x, y, e, marker='+',elinewidth=0.2,markersize=0.5, color=color)

	axs[0,1].loglog(x0_15,fit15,linestyle='--',color='k',linewidth=0.5,label=label0_15+label1_15)
	axs[0,1].loglog(x0_15_ol,fit15_ol,linestyle =':',color='grey',label = label2_15)
	axs[0,1].loglog(x0_15,fit15_p,linestyle='--',color='k',linewidth=0.5)
	axs[0,1].loglog(x0_15,fit15_m,linestyle='--',color='k',linewidth=0.5)
	#axs[0,1].loglog(x0_15,schlauch15_p,linestyle=':',color='k',linewidth=0.5)
	#axs[0,1].loglog(x0_15,schlauch15_m,linestyle=':',color='k',linewidth=0.5)

	axs[0,1].fill_between(x0_15,fit15_p,fit15_m,color = 'gainsboro')
	#axs[0,1].fill_between(x0_15,schlauch15_p,schlauch15_m,color = 'lightpink')
	axs[0,1].set_xlim(10**(-1),10**4)
	axs[0,1].yaxis.tick_right()
	axs[0,1].set_ylim(np.min(y43)*(1-0.1),np.max(y15)+1)
	axs[0,1].legend(loc=2,prop={'size': 3})
	axs[0,0].set_yscale('log')
	axs[0,0].set_xscale('log')
	
	## plot 15 dispersion
	axs[1,1].axvspan(np.min(x15), np.max(x15[mask15]), alpha=0.5, color='grey')

	for x, y, color in zip(x15, Dchisquared(logy15,np.log10(amp15*x15**index15),logyerr15), ep_color15):
		#axs[0,0].plot(x, y, 'o', color=color)
		#axs[1,0].errorbar(x, y, e, marker='+',elinewidth=0.2,markersize=0.5, color=color)
		axs[1,1].loglog(x,y,marker='+',linewidth=0,color = color,markersize=2)
	#axs[1,1].loglog(x15,Dchisquared(logy15,np.log10(amp15*x15**index15),logyerr15),marker='+',linewidth=0,color='R',markersize=2)
	axs[1,1].set_xscale('log')
	axs[1,1].set_xlim(10**(-1),10**4)
	axs[1,1].yaxis.tick_right()

	
	fig.tight_layout()
	plt.subplots_adjust(hspace=0)
	plt.subplots_adjust(wspace=0.02)
	
	fig.add_subplot(111, frameon=False)
	plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
	plt.xlabel(xlab)
	
	plt.savefig(f'./paper_plots_pc_deproj/{Name[source_param]}+_{jet_param}.pdf',dpi=500)
	plt.close()
	#0 index15,1 amp15,2 error15,3 chisq_red15,4 residual15,5 index43,6 amp43,
	#7 error43,8 chisq_red43,9 residual43,10 index15_ol,11 amp15_ol,12 error15_ol,
	#13 chisq_red15_ol,14 residual15_ol,15 index43_ol,16 amp43_ol,17 error43_ol,18 chisq_red43_ol,19 residual43_ol
	return(index15,amp15,error15,chisq_red15,residual15,index43,amp43,error43,chisq_red43,residual43,index15_ol,amp15_ol,error15_ol,chisq_red15_ol,residual15_ol,index43_ol,amp43_ol,error43_ol,chisq_red43_ol,residual43_ol)

l = []
s = []
#sd = []
for i in tqdm.tqdm(range(len(Name))):
	l.append([plot_gradients(i,'l')])
	s.append([plot_gradients(i,'s')])
	#sd.append([plot_gradients(i,'sd')])
l=np.array(l,dtype='object')
s=np.array(s,dtype='object')
#sd=np.array(sd,dtype='object')
np.save('./numpy_arrays/l_pc_(de)proj_test.npy',l)
np.save('./numpy_arrays/s_pc_(de)proj_test.npy',s)
#np.save('./numpy_arrays/sd_pc_(de)proj_test.npy',sd)











