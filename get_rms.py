from astropy.io import fits
import glob
import numpy as np
import os
import tqdm


path = os.getcwd()
names = np.load(path+'/24_03_20/numpy_arrays/list_info/names.npy')

def theta_lim(beam,flux,noise):
	factor = 4*np.log(2)/np.pi
	function = np.log(flux/(flux-noise))
	return(beam*np.sqrt(factor*function))

#file = open('./testfile_15.txt','w') 
 
def get_rms(freq):
	df = []
	for i in tqdm.tqdm(names):
	#for i in names:
		#print(i)
		df_sub = []
		path_temp = path+'/'+str(freq)+'GHz/'+str(i)+'/'+str(i)+'circ_fits/'
		for j in tqdm.tqdm(glob.glob(path_temp+'*.fits')):
			df_sub.append(fits.open(j)[0].header['Noise']*np.pi*fits.open(j)[0].header['BMAJ']*fits.open(j)[0].header['BMIN']*(3600*1000)**2)
			#print(i,fits.open(j)[0].header['DATE-OBS'],fits.open(j)[0].header['Noise'])
			#file.write(str([i,fits.open(j)[0].header['DATE-OBS'],fits.open(j)[0].header['Noise']])+str('\\'))
	#file.close() 

		df.append(df_sub)
	return(np.array(df))

rms_43 = get_rms(43)
#rms43 = np.array(rms_43)
np.save('rms_43.npy',rms_43)

rms_15 = get_rms(15)

#print(rms_15)

np.save('rms_15.npy',rms_15)
