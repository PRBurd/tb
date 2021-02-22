import numpy as np
from astropy.io import fits
from scipy import optimize
from scipy.odr import ODR,Model,RealData
import matplotlib.pyplot as plt
import os
from numpy.random import choice
import tqdm

Name = np.load('./../numpy_arrays/list_info/names.npy')

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
#0A,1x_break,2a1,3a2,4scaling
def broken_powerlaw(beta,x):
    temp = beta[0] * (x/beta[1])**(-beta[2]) * (1/2*(1+x/beta[1])**(1/beta[4]))**((beta[2]-beta[3])*beta[4])
    return(temp)
def power_law(beta,x):
    temp = beta[0]*x**beta[1]
    return(temp)
#0A,1x_break,2a1,3a2,4scaling
#5 break 2, scaling 2, 7 slope 3 , 8 break 3, 9 scaling 3, 10 slope 4
def _3broken(beta,x):
    temp1 = beta[0] * (x/beta[1])**(-beta[2]) * (1/2*(1+x/beta[1])**(1/beta[4]))**((beta[2]-beta[3])*beta[4])
    temp2 = (1/2*(1+x/beta[5])**(1/beta[6]))**((beta[3]-beta[7])*beta[6])
    temp3 = (1/2*(1+x/beta[8])**(1/beta[9]))**((beta[7]-beta[10])*beta[9])
    return(temp1*temp2*temp3)
def _5broken(beta,x):
    temp1 = beta[0] * (x/beta[1])**(-beta[2]) * (1/2*(1+x/beta[1])**(1/beta[4]))**((beta[2]-beta[3])*beta[4])
    temp2 = (1/2*(1+x/beta[5])**(1/beta[6]))**((beta[3]-beta[7])*beta[6])
    temp3 = (1/2*(1+x/beta[8])**(1/beta[9]))**((beta[7]-beta[10])*beta[9])
    
    temp4 = (1/2*(1+x/beta[11])**(1/beta[12]))**((beta[10]-beta[13])*beta[12])
    temp5 = (1/2*(1+x/beta[14])**(1/beta[15]))**((beta[13]-beta[16])*beta[15])
    return(temp1*temp2*temp3*temp4*temp5)
def ODR_fitting(xdata,ydata,fitfunction,beta,fix):
    bpl_all = Model(fitfunction)
    data_all = RealData(xdata, ydata, sx=np.cov([xdata,ydata])[0][1], sy=np.cov([xdata,ydata])[0][1])
    odr_all = ODR(data_all, bpl_all, beta0=beta,ifixb=fix)
    odr_all.set_job(fit_type=0)
    output_all = odr_all.run()
    #output_all.pprint()
    return(output_all.beta,output_all.sd_beta)
def MC_broken(xb_min,xb_max,data_x,data_y,data_err):
    #break regions
    x_breaks = np.linspace(xb_min,xb_max,10000)
    betas_in = []
    betas_out = []
    x_break = []
    betas_inerr = []
    betas_outerr = []
    for i in tqdm.tqdm(range(0,5000)):
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



Name = np.load('./../numpy_arrays/list_info/names.npy')
for i in range(len(Name)):
	x,y,err = load_data(i,'l')
	lout,lin,x_break,louterr,linerr = MC_broken(np.sort(x)[10],np.sort(x)[-10],x,y,err)
	np.save('./possible_breaks/{}_lout.npy'.format(Name[i]),lout)
	np.save('./possible_breaks/{}_lin.npy'.format(Name[i]),lin)
	np.save('./possible_breaks/{}_xbreak.npy'.format(Name[i]),x_break)
	np.save('./possible_breaks/{}_louterr.npy'.format(Name[i]),louterr)
	np.save('./possible_breaks/{}_linerr.npy'.format(Name[i]),linerr)
#x,y,err = load_data(0,'l')

#lout,lin,x_break,louterr,linerr = MC_broken(np.sort(x)[10],np.sort(x)[-10],x,y,err)


    

