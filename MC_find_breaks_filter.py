import numpy as np
from astropy.io import fits
from scipy import optimize
from scipy.odr import ODR,Model,RealData
import matplotlib.pyplot as plt
import os
from numpy.random import choice
import tqdm

def get_data(freq,index):
	pwd = os.getcwd()
	Name = np.load('./../numpy_arrays/list_info/names.npy')
	print(Name[index]+' {} GHz'.format(freq))
	data = np.load(pwd+'/../data_test/'+str(Name[index])+'_'+str(freq)+'GHz.npy')
	return(data)
def get_parameters(source_param,jet_param):
	d43 = get_data(43,source_param)
	d15 = get_data(15,source_param)
	
	if jet_param == 'l':
		x43 = d43[0]
		y43 = d43[1] 
		x15 = d15[0]
		y15 = d15[1]
		pinit = [1.0,1.0]
		mask15 = np.where(x15 <= np.max(x43))[0]
		mask43 = np.where(x43 >= np.min(x15))[0]
		err43 = d43[5]
		err15 = d15[5]
		jp = 'l'
		ylab = r'$d_j/\mathrm{mas}$'
		xlab = r'$d_c/\mathrm{mas}$'

	if jet_param == 's':
		x43 = d43[0]
		y43 = d43[2] 
		x15 = d15[0]
		y15 = d15[2]  
		pinit = [1.0,-1.0]
		mask15 = np.where(x15 <= np.max(x43))[0]
		mask43 = np.where(x43 >= np.min(x15))[0]
		err43 = d43[4]
		err15 = d15[4]
		jp = 's'
		ylab = r'$t_b/\mathrm{K}$'
		xlab = r'$d_c/\mathrm{mas}$'

	if jet_param == 'sd':
		x43 = d43[1]
		y43 = d43[2] 
		x15 = d15[1]
		y15 = d15[2]
		pinit = [1.0,-1.0]
		mask15 = np.where(x15 <= np.max(x43))[0]
		mask43 = np.where(x43 >= np.min(x15))[0]
		err43 = d43[4]
		err15 = d15[4]
		jp=r'$s_d$'
		ylab = r'$t_b/\mathrm{K}$'
		xlab = r'$d_j/\mathrm{mas}$'

	epoch15 = d15[3]
	epoch43 = d43[3]
	return(x15,y15,err15,x43,y43,err43)
def load_data(index,power_law_index):
	data = np.array(get_parameters(index,power_law_index))
	x = np.append(data[0],data[3])
	y = np.append(data[1],data[4])
	err =  np.append(data[2],data[5])
	return(x,y,err)    
def power_law(beta,x):
	temp = beta[0]*x**beta[1]
	return(temp)

def MC_broken(xb_min,xb_max,data_x,data_y,data_err):
	#break regions
	x_breaks = np.linspace(xb_min,xb_max,10000)
	betas_in = []
	betas_out = []
	x_break = []
	betas_inerr = []
	betas_outerr = []
	for i in tqdm.tqdm(range(0,10000)):
		i+=1
		xb = choice(x_breaks)
		mask0 = np.where(data_x <xb,True,False)
		mask1 = np.where(data_x >=xb,True,False)
		#fitting
		betas1 = ODR_fitting(data_x[mask1],data_y[mask1],power_law,np.ones(shape=2),np.ones(shape=2))[0]
		betas0 = ODR_fitting(data_x[mask0],data_y[mask0],power_law,[1,1],[1,1])[0]
		betas0_err = ODR_fitting(data_x[mask0],data_y[mask0],power_law,np.ones(shape=2),np.ones(shape=2))[1]
		betas1_err = ODR_fitting(data_x[mask1],data_y[mask1],power_law,np.ones(shape=2),np.ones(shape=2))[1]
		betas_in.append(betas1)
		betas_out.append(betas0)
		betas_inerr.append(betas1_err)
		betas_outerr.append(betas0_err)
		x_break.append(xb)
	return(np.array(betas_in).transpose(1,0),np.array(betas_out).transpose(1,0),np.array(x_break),np.array(betas_outerr).transpose(1,0),np.array(betas_inerr).transpose(1,0))
def ODR_fitting(xdata,ydata,fitfunction,beta,fix):
    bpl_all = Model(fitfunction)
    data_all = RealData(xdata, ydata, sx=np.cov([xdata,ydata])[0][1], sy=np.cov([xdata,ydata])[0][1])
    odr_all = ODR(data_all, bpl_all, beta0=beta,ifixb=fix)
    odr_all.set_job(fit_type=0)
    output_all = odr_all.run()
    #output_all.pprint()
    return(output_all.beta,output_all.sd_beta)

Name = np.load('./../numpy_arrays/list_info/names.npy')

#for i in range(len(Name)):
for i in range(len(Name)):
	x,y,err = load_data(i,'l')
	lout,lin,x_break,louterr,linerr = MC_broken(np.sort(x)[5],np.sort(x)[-5],x,y,err)
	mask = np.where(lout[1]/lin[1]>1,True,False)*np.where(lout[1]-lin[1]>0,True,False)*np.where(lout[1] > lin[1],True,False)*np.where(linerr[0]/louterr[0]<10,True,False)*np.where(linerr[0]/louterr[0]>0,True,False)*np.where(linerr[1]/louterr[1]<10,True,False)*np.where(linerr[1]/louterr[1]>0,True,False)
	print(mask)
	lin_mean = np.mean(lin[1][mask])
	lout_mean = np.mean(lout[1][mask])
	x_break_med = np.median(x_break[mask])
	print(lin,lout,x_break)
	plot_data = get_parameters(i,'l')
	_43y = plot_data[4]
	_43y_err = plot_data[5]
	_43x = plot_data[3]
	_15y = plot_data[1]
	_15y_err = plot_data[2]
	_15x = plot_data[0]

	plt.errorbar(_43x,_43y,yerr =_43y_err, color = 'grey', label=str(Name[i])+' 43 GHz, lin={}'.format(np.round(lin_mean,2)),linewidth=0,elinewidth = 0.5,marker = 'x')
	plt.errorbar(_15x,_15y,yerr =_15y_err, color = 'grey', label=str(Name[i])+' 15 GHz,lout={}'.format(np.round(lout_mean,2)),linewidth=0,elinewidth = 0.5,marker = '^')
	plt.vlines(x_break_med,min(y),max(y),color = 'k',label='x_break = {}'.format(np.round(x_break_med,2)))
	plt.xscale('log')
	plt.yscale('log')
	for j in tqdm.tqdm(range(len(lin[0][mask]))):
		xin = np.linspace(np.min(x),np.max(x[np.where(x < x_break[j],True,False)]),10000)
		xout = np.linspace(np.min(x[np.where(x >= x_break[j],True,False)]),np.max(x),10000)

		mask0 = np.where(x < x_break[j],True,False)
		mask1 = np.where(x >= x_break[j],True,False)
		temp0 = (y[mask0]- lin[0][mask][j]*x[mask0]**lin[1][j])**2/err[mask0]
		temp1 = (y[mask1]- lout[0][mask][j]*x[mask1]**lout[1][j])**2/err[mask1]
		rchi2 = sum(temp0)/(len(temp0)-2) + sum(temp1)/(len(temp1)-2)

		if rchi2 > 0.8 and rchi2 < 4 and len(temp1)/len(temp0)>0.2:
			plt.loglog(xin[mask],lin[0][mask][j]*xin[mask]**lin[1][mask][j],color='blue',alpha=0.1,linewidth=0.5,zorder=9999)
			plt.loglog(xout[mask],lout[0][mask][j]*xout[mask]**lout[1][mask][j],color='red',alpha=0.1,linewidth=0.5,zorder=9999)
			plt.xlabel(r'$d_c$')
			plt.ylabel(r'$d_j$')
			plt.legend()
		else:
			pass
	plt.savefig('./geometry_breaks_plots/{}_bpl.pdf'.format(Name[i]))
	plt.close()
	






