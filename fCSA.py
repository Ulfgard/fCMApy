import numpy as np
import math 
class fCSA:
	def __init__(self, init_mean, init_variance = 1.0, noise_adaptation=False):
		#variables of the normal distribution
		self.mean = init_mean
		self.variance = init_variance
		
		#integer variables for population and dimensionality
		self.n = init_mean.shape[0]
		self.n_off = int( 4. + math.floor( 3 * math.log(self.n )))
		
		#learning rate and initial value for mu_eff
		self._mu_eff = float(self.n_off)
		self._cmu_eff = 0.01
		
		#variables for CSA
		self._path=np.zeros(self.n)
		self._gamma_path = 0.0
		
		#noise adaptation
		self.rate = 1.0
		self.noise_adaptation = noise_adaptation
		self._fvar = 1.0
		self._sigma_noise = 1.0
	
		self.avg_loss=0.0
		
		
	def step(self, function):
		# generate offspring function
		sigma = math.sqrt(self.variance);
		z = np.random.normal(np.zeros((self.n_off, self.n)))
		x = self.mean[np.newaxis,:] + sigma * z
			
		# evaluate offspring
		fvals=np.zeros(self.n_off)
		for i in range(self.n_off):
			fvals[i] = function(x[i,:])
			
			
		#noise handling
		if self.noise_adaptation:
			#acquire second evaluation
			fvals2=np.zeros(self.n_off)
			for i in range(self.n_off):
				fvals2[i] = function(x[i,:])
			
			#gather noise variables
			var_noise = np.mean((fvals-fvals2)**2)
			fvals = (fvals+fvals2)/2
			fmean = np.mean(fvals)
			fvar = np.sum((fvals - fmean)**2)/(self.n_off -1.0)
			
			
			#update noise statistics
			cztest = 0.01 * self.rate**1.5;
			self._fvar = (1-cztest)*self._fvar + cztest * fvar;
			self._sigma_noise = (1-cztest) * self._sigma_noise + cztest * var_noise;
			ztest = 0.5*(self._fvar/self._sigma_noise - 1);
			self.rate = 1.0/(1.0+1.0/(ztest/self.n_off));
			
		#store estimate for current loss
		self.avg_loss= np.mean(fvals)
			
		self._update(x,z,fvals)
	
	def _update(self, x,z,fvals):
		#compute normalized function values
		weights = -fvals + max(fvals)
		weights /= np.sum(np.abs(weights))
		
		#compute individual learning-rates
		cPath = 2*(self._mu_eff + 2.)/(self.n + self._mu_eff + 5.) * self.rate
		damping_path = 2 * self.rate * cPath/math.sqrt(cPath * (2-cPath))
		
		#compute gradient of mean and normalized step-length
		dMean = np.sum((weights - 1.0/self.n_off)[:,np.newaxis] * x, axis = 0)
		step_z = np.sum(weights[:,np.newaxis] * z, axis = 0)
		
		#update evolution-path
		self._path = (1-cPath) * self._path + math.sqrt(cPath * (2-cPath) * self._mu_eff) * step_z;
		self._gamma_path = (1-cPath)**2 * self._gamma_path + cPath * (2-cPath);
		deviation_step_len = math.sqrt(np.mean(self._path**2)) - math.sqrt(self._gamma_path);
		
		#update evariables
		self.mean += self.rate * dMean;
		self.variance *= np.exp(deviation_step_len*damping_path);
		self._mu_eff = (1-self._cmu_eff) * self._mu_eff + self._cmu_eff / np.sum(weights**2);