import numpy as np


if __name__ == '__main__':
	print('bitte als library irgendwo einbinden')
else:
	print('successfully loaded')



class magnetic_fields:
	def __init__(self,s=-2.5,l=1,alpha=-0.75,M=5*10**8,j=0.5, P=10**(44),B = 1):
		self.s = s
		self.l = l
		self.alpha = alpha
		self.M = M 
		self.j = j
		self.P = P
		self.B = B


	@property
	def b_equip(self):
		A = self.s-self.l
		B = 3-self.alpha
		return(A/B)
	@property
	def b_pcons(self):
		A = self.s+self.l
		B = 1-self.alpha
		return(A/B)

	def db_equip(self,ds,dl,dalpha):
		alpha = self.alpha
		s = self.s
		l = self.l
		A = 1/(2-alpha) * ds
		B = -1/(2-alpha) * dl
		C = (s-l)/(2-alpha)**2 * dalpha
		D = np.abs(A * B)
		err = np.sqrt(A**2+B**2+C**2+D)
		self.db_eq = err
		return(err)

	def db_pcons(self,ds,dl,dalpha):
		alpha = self.alpha
		s = self.s
		l = self.l
		A = 1/(2-alpha) * ds
		B = 1/(2-alpha) * dl
		C = (s+l)/(2-alpha)**2 * dalpha
		D = np.abs(A * B)
		err = np.sqrt(A**2+B**2+C**2+D)
		self.db_pc = err
		return(err)

	@property
	def B_eddington(self):
		return(10**4 * 6*np.sqrt(self.M/10**8))
	@property 
	def B_spin_BZ(self):
		j = self.j
		M = self.M
		P = self.P
		return(np.sqrt(5)/j *(P/10**(44))**0.5 * (M/10**8)**(-1))
	@property
	def spin_BZ(self):
		B = self.B
		P = self.P
		M = self.M 
		return(np.sqrt(5)*(P/10**(44))**(0.5)*(B/10**4)**(-1)*(M/10**8)**(-1))

	def dist(self,B0,b):
		B = self.B
		r = (B/B0)**(1/b)
		return(r)
	
	def ddist(self,B0,b,db):
		B = self.B
		a = B/B0
		a = np.array(a)
		temp = np.array(((1/b**2 * np.log(a) * i**(1/b) * db)**2)**0.5)
		return(temp)
	@property
	def grav_rad(self):
		M = self.M
		c = 299792458
		G = 6.67430*10**(-11)
		mass = M * 1.98847*10**(30)
		r = G*mass/c**2
		return(r)