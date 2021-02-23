import numpy as np 
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.stats import linregress as linreg
from scipy.stats import chisquare as chisq
import os
import shutil




def adjustFigAspect(fig,aspect=1):
    '''
    Adjust the subplot parameters so that the figure has the correct
    aspect ratio.
    '''
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

def linfit(x,y):
    slope,intercept,rvalue,pvalue,stderr = linreg(np.log(x),np.log(y))
    fit = (x**(slope) * np.exp(intercept))
    return(fit,slope,intercept,rvalue,pvalue,stderr)
name = [
'0219+428',
'0336-019',
'0415+379',
'0430+052',
'0528+134',
'0716+714',
'0735+178',
'0827+243',
'0829+046',
'0836+710',
'0851+202',
'0954+658',
'1101+384',
'1127-145',
'1156+295',
'1219+285',
'1222+216',
'1226+023',
'1253-055',
'1308+326',
'1633+382',
'1652+398',
'1730-130',
'1749+096',
'2200+420',
'2223-052',
'2230+114',
'2251+158',
]
co_name = [
'3C 66A',
'CTA 26',
'3C111', 
'3C 120', 
'PKS 0528+134',
'TXS 0716+714',
'OI 158',
'OJ 248',
'OJ 049',
'4C +71.07',
'OJ 287',
'S4 0954+65',
'Mrk 421',
'PKS 1127-14',
'4C +29.45',
'W Comae',
'4C +21.35',
'3C 273',
'3C 279',
'OP 313',
'4C +38.41',
'Mrk 501',
'NRAO 530',
'OT 081',
'BL Lac',
'3C 446',
'CTA 102',
'3C 454.3',
]    
dst = '/scratch/local/Paul/VLBA/fitpaul_backup_010715/'
src15 = '/scratch/local/Paul/VLBA/fitpaul_backup_010715/15_GHz_fits/' 
src43 = '/scratch/local/Paul/VLBA/fitpaul_backup_010715/43_GHz_fits/' 

src15_array = []
src43_array = []

for i in range(len(name)):
	src15_array.append(src15 + name[i] + '_15GHz_diameter.fits')
	src43_array.append(src43 + name[i] + '_43GHz_diameter.fits')


for i in range(len(name)):
	savenp15 = name[i]+ '_15GHz' + '.npy'
	savenp43 = name[i]+ '_43GHz' + '.npy'
	#print(i)
	#open fits data
	data15 = fits.open(src15_array[i])
	data43 = fits.open(src43_array[i])
	#set common name
	c_name = co_name[i]

	#np_array the 15 GHz data
	data15_np = np.array(data15[1].data)
	#get c_ditance, maj_ax and tb 
	c_distance15 = []
	for i in range(len(data15_np)):
		c_distance15.append(data15_np[i][25])
	c_distance15 = np.array(c_distance15)
	maj_ax15 = []
	for i in range(len(data15_np)):
		maj_ax15.append(data15_np[i][3])
	maj_ax15= np.array(maj_ax15)
	comp15 = []
	for i in range(len(data15_np)):
		comp15.append(data15_np[i][14])
	comp15 = np.array(comp15)
	tb15 = []
	for i in range(len(data15_np)):
		tb15.append(data15_np[i][11])
	tb15 = np.array(tb15)
	dia_plot15 = np.concatenate((comp15,c_distance15,maj_ax15,tb15))
	dia_plot15 = np.reshape(dia_plot15,(4,-1))
### plot parameters l vs. c_dist
	c_d15 = []
	m_a15 = []
	t_b15 = []
	for i in range(len(dia_plot15[0])):
		if dia_plot15[0][i] == 0:
			c_d15.append(dia_plot15[1][i])
			m_a15.append(dia_plot15[2][i])
			t_b15.append(dia_plot15[3][i])
	c_d15 = np.array(c_d15)
	m_a15 = np.array(m_a15)
	t_b15 = np.array(t_b15)
	dia15 = np.concatenate((c_d15,m_a15,t_b15))
	dia15 = np.reshape(dia15,(3,-1))
	try:
		os.remove(savenp15)
	except OSError:
		pass
	np.save(savenp15,dia15)




	data43_np = np.array(data43[1].data)
	c_distance43 = []
	for i in range(len(data43_np)):
		c_distance43.append(data43_np[i][25])
	c_distance43 = np.array(c_distance43)
	maj_ax43 = []
	for i in range(len(data43_np)):
		maj_ax43.append(data43_np[i][3])
	maj_ax43= np.array(maj_ax43)
	comp43 = []
	for i in range(len(data43_np)):
		comp43.append(data43_np[i][14])
	comp43 = np.array(comp43)
	tb43 = []
	for i in range(len(data43_np)):
		tb43.append(data43_np[i][11])
	tb43 = np.array(tb43)
	dia_plot43 = np.concatenate((comp43,c_distance43,maj_ax43,tb43))
	dia_plot43 = np.reshape(dia_plot43,(4,-1))
### plot parameters l vs. c_dist
	c_d43 = []
	m_a43 = []
	t_b43 = []
	for i in range(len(dia_plot43[0])):
		if dia_plot43[0][i] == 0:
			c_d43.append(dia_plot43[1][i])
			m_a43.append(dia_plot43[2][i])
			t_b43.append(dia_plot43[3][i])
	c_d43 = np.array(c_d43)
	m_a43 = np.array(m_a43)
	t_b43 = np.array(t_b43)
	dia43 = np.concatenate((c_d43,m_a43,t_b43))
	dia43 = np.reshape(dia43,(3,-1))
	try:
		os.remove(savenp43)
	except OSError:
		pass
	np.save(savenp43,dia43)


 
'''
building np.arrays with all fit parameters, necessary for plotting
'''
for i in range(len(name)):
	savefit = dst + name[i] 


	temp15 = dst + name[i] + '_15GHz.npy'
	fit15 = np.load(temp15)
	# fit l
	fit_l15,slope_l15,intercept_l15,rvalue_l15,pvalue_l15,stderr_l15 = linfit(fit15[0],fit15[1])
	#fit tb
	fit_tb15,slope_tb15,intercept_tb15,rvalue_tb15,pvalue_tb15,stderr_tb15 = linfit(fit15[0],fit15[2])
	#fit sd
	fit_sd15,slope_sd15,intercept_sd15,rvalue_sd15,pvalue_sd15,stderr_sd15 = linfit(fit15[1],fit15[2])

	l15 = []
	l15.append((fit_l15,slope_l15,rvalue_l15,stderr_l15))
	l15 = np.array(l15)
	l15 = np.reshape(l15,-1)

	tb15 = []
	tb15.append((fit_tb15,slope_tb15,rvalue_tb15,stderr_tb15))
	tb15 = np.array(tb15)
	tb15 = np.reshape(tb15,-1)

	sd15 = []
	sd15.append((fit_sd15,slope_sd15,rvalue_sd15,stderr_sd15))
	sd15 = np.array(sd15)
	sd15 = np.reshape(sd15,-1)

	temp43 = dst + name[i] + '_43GHz.npy'
	fit43 = np.load(temp43)
	# fit l
	fit_l43,slope_l43,intercept_l43,rvalue_l43,pvalue_l43,stderr_l43 = linfit(fit43[0],fit43[1])
	#fit tb
	fit_tb43,slope_tb43,intercept_tb43,rvalue_tb43,pvalue_tb43,stderr_tb43 = linfit(fit43[0],fit43[2])
	#fit sd
	fit_sd43,slope_sd43,intercept_sd43,rvalue_sd43,pvalue_sd43,stderr_sd43 = linfit(fit43[1],fit43[2])

	l43 = []
	l43.append((fit_l43,slope_l43,rvalue_l43,stderr_l43))
	l43 = np.array(l43)
	l43 = np.reshape(l43,-1)


	tb43 = []
	tb43.append((fit_tb43,slope_tb43,rvalue_tb43,stderr_tb43))
	tb43 = np.array(tb43)
	tb43 = np.reshape(tb43,-1)


	sd43 = []
	sd43.append((fit_sd43,slope_sd43,rvalue_sd43,stderr_sd43))
	sd43 = np.array(sd43)
	sd43 = np.reshape(sd43,-1)

	try:
		os.remove(savefit + '_fit_tb_15.npy')
		os.remove(savefit + '_fit_l_15.npy')
		os.remove(savefit + '_fit_sd_15.npy')
		os.remove(savefit + '_fit_tb_43.npy')
		os.remove(savefit + '_fit_l_43.npy')
		os.remove(savefit + '_fit_sd_43.npy')
	except OSError:
		pass
	np.save(savefit + '_fit_tb_15.npy',tb15)
	np.save(savefit + '_fit_l_15.npy',l15)
	np.save(savefit + '_fit_sd_15.npy',sd15)
	np.save(savefit + '_fit_tb_43.npy',tb43)
	np.save(savefit + '_fit_l_43.npy',l43)
	np.save(savefit + '_fit_sd_43.npy',sd43)

