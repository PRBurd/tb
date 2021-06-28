import numpy as np 
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.stats import linregress as linreg
from scipy.stats import chisquare as chisq
import os
import shutil
import os
import tqdm


def theta_lim(beam,flux,noise):
	factor = 4*np.log(2)/np.pi
	function = np.log(flux/(flux-noise))
	return(beam*np.sqrt(factor*function))

def sigma_peak(rms,flux):
	err = np.sqrt((rms*(1+flux/rms)**(0.5))**2+(0.05*flux)**2)
	return(err)
		   
def sigma_dia(dia,sigpeak,flux,beam):
	errdia = np.sqrt(dia*(sigpeak/flux)**2)# +(0.1*beam)**2)
	#errdia = 0.18*dia
	return(errdia)

def sigma_dist(sigdia):
	return(sigdia*0.5)

def sigma_tb(dia,fluxJy,ddia,dfluxJy,frequGHz,z,tb):
	c0 = 299792458
	kb = 1.380649*10**(-23)
	#dia = 10**3 * dia
	#ddia = 10**3 * ddia
	flux = fluxJy *10**(-26)
	dflux = dfluxJy * 10**(-26)
	dia = dia *((1/3600000*np.pi/180))
	ddia = ddia * ((1/3600000*np.pi/180))
	A = (2*np.log(2)/(np.pi*kb)) *(c0/(frequGHz*10**9))**2 *(1+z) #*10**(-26) /((1/3600000*np.pi/180))**2
	errtb = ((A*(1/(dia)**2)*dflux)**2+((2*A*flux/((dia)**3))*(ddia))**2)**0.5
	errtb = np.sqrt(errtb**2)#+ (0.19 * tb)**2)
	#errtb = 0.44 * tb
	return(errtb)

path = os.getcwd()
name = np.load(path+'/24_03_20/numpy_arrays/list_info/names.npy')
redshift = np.load(path+'/24_03_20/numpy_arrays/list_info/z.npy')


src15 = os.getcwd()+'/15_GHz_fits/'
src43 = os.getcwd()+'/43_GHz_fits/'

rms_15 = np.load(os.getcwd()+'/24_03_20/numpy_arrays/list_info/rms_15_fill.npy')
rms_43 = np.load(os.getcwd()+'/24_03_20/numpy_arrays/list_info/rms_43_fill.npy')



for i in tqdm.tqdm(range(len(name))):
	fits15 = src15 + name[i] + '_15GHz_diameter.fits'
	fits43 = src43 + name[i] + '_43GHz_diameter.fits'
	df15 = fits.open(fits15)
	df43 = fits.open(fits43)


	try:
		print(name[i])
		lim15 = theta_lim(df15[1].data['bmaj'],df15[1].data['flux'],df15[1].data['noise'])
		lim43 = theta_lim(df43[1].data['bmaj'],df43[1].data['flux'],df43[1].data['noise'])
		#lim15 = theta_lim(df15[1].data['bmaj'],df15[1].data['flux'],df15[1].data['noise'])
		#lim43 = theta_lim(df43[1].data['bmaj'],df43[1].data['flux'],df43[1].data['noise'])

		sigma_flux15 = sigma_peak(df15[1].data['noise']*np.pi*df15[1].data['bmaj']*df15[1].data['bmin'],df15[1].data['flux'])
		sigma_flux43 = sigma_peak(df43[1].data['noise']*np.pi*df43[1].data['bmaj']*df43[1].data['bmin'],df43[1].data['flux'])
		#sigma_flux15 = sigma_peak(0.4*10**(-3),df15[1].data['flux'])
		#sigma_flux43 = sigma_peak(0.4*10**(-3),df43[1].data['flux'])

		# is b min radius or diameter
		sigma_dia15 = sigma_dia(df15[1].data['major_ax'],sigma_flux15,df15[1].data['flux'],np.pi*(df15[1].data['bmaj']+df15[1].data['bmin'])/2)
		sigma_dia43 = sigma_dia(df43[1].data['major_ax'],sigma_flux43,df43[1].data['flux'],np.pi*(df43[1].data['bmaj']+df43[1].data['bmin'])/2)

		sigma_tb15 = sigma_tb(df15[1].data['major_ax'],df15[1].data['flux'],sigma_dia15,sigma_flux15,15,redshift[i],df15[1].data['tb'])
		sigma_tb43 = sigma_tb(df43[1].data['major_ax'],df43[1].data['flux'],sigma_dia43,sigma_flux43,43,redshift[i],df43[1].data['tb'])

		sigma_dist15 = 0.5*sigma_dia15
		sigma_dist43 = 0.5*sigma_dia43	
	
		mask15 = np.where(df15[1].data['major_ax'] >lim15,True,False)*np.where(df15[1].data['comp'] == 0,True,False)
		mask43 = np.where(df43[1].data['major_ax'] >lim43,True,False)*np.where(df43[1].data['comp'] == 0,True,False)

		df_15temp = np.array([df15[1].data['c_distance'][mask15],df15[1].data['major_ax'][mask15],df15[1].data['tb'][mask15],df15[1].data['ep'][mask15],sigma_tb15[mask15],sigma_dia15[mask15],sigma_dist15[mask15]])
		df_43temp = np.array([df43[1].data['c_distance'][mask43],df43[1].data['major_ax'][mask43],df43[1].data['tb'][mask43],df43[1].data['ep'][mask43],sigma_tb43[mask43],sigma_dia43[mask43],sigma_dist43[mask43]])

		np.save(path+'/24_03_20/data_test/'+name[i]+'_15GHz.npy',df_15temp)
		np.save(path+'/24_03_20/data_test/'+name[i]+'_43GHz.npy',df_43temp)
	except:
		print(name[i])
		mask15 = np.where(df15[1].data['comp'] == 0,True,False)
		mask43 = np.where(df43[1].data['comp'] == 0,True,False)

		sigma_flux15 = sigma_peak(0.4*10**(-3),df15[1].data['flux'])
		sigma_flux43 = sigma_peak(0.4*10**(-3),df43[1].data['flux'])

		sigma_dia15 = sigma_dia(df15[1].data['major_ax'],sigma_flux15,df15[1].data['flux'],np.pi*(df15[1].data['bmaj']+df15[1].data['bmin'])/2)
		sigma_dia43 = df43[1].data['major_ax']*df43[1].data['rel_maj_err']

		sigma_tb15 = sigma_tb(df15[1].data['major_ax'],df15[1].data['flux'],sigma_dia15,sigma_flux15,15,redshift[i],df15[1].data['tb'])
		sigma_tb43 = df43[1].data['tb']*df43[1].data['rel_tb_err']

		sigma_dist15 = 0.5*sigma_dia15
		sigma_dist43 = 0.5*sigma_dia43	

		df_15temp = np.array([df15[1].data['c_distance'][mask15],df15[1].data['major_ax'][mask15],df15[1].data['tb'][mask15],df15[1].data['ep'][mask15],sigma_tb15[mask15],sigma_dia15[mask15],sigma_dist15[mask15]])
		df_43temp = np.array([df43[1].data['c_distance'][mask43],df43[1].data['major_ax'][mask43],df43[1].data['tb'][mask43],df43[1].data['ep'][mask43],sigma_tb43[mask43],sigma_dia43[mask43],sigma_dist43[mask43]])

		np.save(path+'/24_03_20/data_test/'+name[i]+'_15GHz.npy',df_15temp)
		np.save(path+'/24_03_20/data_test/'+name[i]+'_43GHz.npy',df_43temp)
