import numpy as np
import matplotlib.pyplot as plt
import math

np.seterr(invalid='raise')

class OnlineLODA:
    
    def __init__(self, dimension, X=None, n_histograms=100, hist_per_batch =1, batch_size=1000, init=10e-25, memory=5):
        self.d = dimension
        self.batch_size = batch_size
        self.init = init
        self.bins = math.ceil(2*((batch_size*memory)**(1/3))) #Rice rule
        self.n_histograms = n_histograms
        self.hist_per_batch = hist_per_batch
        self.histograms = np.asarray([[init]*(self.bins+2)]*self.n_histograms)
        self.vectors = np.zeros((self.n_histograms, self.d))
        self.maxs = np.zeros((n_histograms,))
        self.mins = np.zeros((n_histograms,))
        
        self.devs = np.zeros((n_histograms, self.d))
        self.means = np.zeros((n_histograms, self.d))
        
        self.memory = memory
        
        self.pointer = 0
        
        self.stored_batches = [] #last #memory batches 
        self.batch_pointer = 0 #always points to the lastly added batch
        
        self.total_batches = 0
        
        
        self.predictions = []
        
        self.first_batch = True

        self.hist_num_of_bin = None

        for i in range(self.n_histograms):
            idxs = np.random.choice(range(self.d), self.d-int(np.sqrt(self.d)), replace=False)
            v = np.random.normal(0,1,self.d)            
            v[idxs] = 0
            self.vectors[i]=v
            
            
        if(X is not None):
            iters = X.shape[0]//self.batch_size
            for batchnum in range(iters):
                
                if(batchnum != iters-1):
                    self.train_on_batch(np.asarray(X[batchnum*self.batch_size : (batchnum+1)*self.batch_size].copy()))
                else:
                    self.train_on_batch(np.asarray(X[batchnum*self.batch_size : ].copy()))
                
        
        
    def train_on_batch(self, X, first_batch=False):
        
        current_pred = None

        if(not self.first_batch):
            current_pred = self.predict(X)
            self.predictions.append(current_pred.copy())
        
        if(len(self.stored_batches)<self.memory):
            self.stored_batches.append(X)
        else:
            self.stored_batches[self.batch_pointer] = X
            self.batch_pointer = (self.batch_pointer + 1) % self.memory
        
        X_new = np.vstack(self.stored_batches)
        means = np.mean(X_new, axis=0)
        devs = np.std(X_new, axis=0)
        X_new = (X_new-means)/(devs+1e-10)
        
        
        for _ in range(self.hist_per_batch):
            if(self.total_batches<self.n_histograms):
                
                
                #Create new hist
                res = np.matmul(X_new,self.vectors[self.total_batches].T)
                maxs = np.amax(res, axis=0)*1.1
                mins = np.amin(res, axis=0)*0.9 #nem updateljuk ezeket, a minje es maxa minden histnek a letrejottekore eldol
                self.maxs[self.total_batches] = maxs
                self.mins[self.total_batches] = mins
                self.devs[self.total_batches] = devs+1e-10
                self.means[self.total_batches] = means
                
                
                binsize = (self.maxs[self.total_batches]-self.mins[self.total_batches])/self.bins
                diff = res-self.mins[self.total_batches]
                self.histograms[self.total_batches] = (np.bincount(np.clip((diff//binsize).astype('int')+1,0, self.bins+1), minlength=self.bins+2))/(min(self.total_batches+1, self.memory)*self.batch_size) + 1e-10
                
                self.total_batches += 1
                
            else:
                
                #Create new projection
                idxs = np.random.choice(range(self.d), self.d-int(np.sqrt(self.d)), replace=False)
                v = np.random.normal(0,1,self.d)            
                v[idxs] = 0
                self.vectors[self.pointer]=v
                
                #Create new hist
                res = np.matmul(X_new,self.vectors[self.pointer].T)
                maxs = np.amax(res, axis=0)*1.1
                mins = np.amin(res, axis=0)*0.9 #nem updateljuk ezeket, a minje es maxa minden histnek a letrejottekore eldol
                self.maxs[self.pointer] = maxs
                self.mins[self.pointer] = mins
                self.devs[self.pointer] = devs+1e-10
                self.means[self.pointer] = means
                
                
                binsize = (self.maxs[self.pointer]-self.mins[self.pointer])/self.bins
                diff = res-self.mins[self.pointer]
                
                self.histograms[self.pointer] = np.bincount(np.clip((diff//binsize).astype('int')+1,0, self.bins+1), minlength=self.bins+2)
                
                self.histograms[self.pointer] /= (min(self.total_batches+1, self.memory) * self.batch_size) 
                self.histograms[self.pointer] += 1e-10
                    
                self.pointer = (self.pointer+1)%self.n_histograms
        
        
        if(self.first_batch):
            current_pred = self.predict(X)
            self.predictions.append(current_pred.copy())
            self.first_batch=False

        return current_pred
                
   
    
    def predict(self,X):
        summ = np.zeros((X.shape[0],))
        
        self.hist_num_of_bin = []
        
        for k in range(min(self.n_histograms, self.total_batches)):

            res = np.matmul((X-self.means[k])/self.devs[k],self.vectors[k].T)

            binsize = (self.maxs[k]-self.mins[k])/self.bins
            diff = res-self.mins[k]

            
            self.hist_num_of_bin.append(np.clip((diff//binsize)+1,0, self.bins+1).astype('int'))
            
            
            summ -= np.log(self.histograms[k][np.clip((diff//binsize)+1,0, self.bins+1).astype('int')])
            
            
        summ = summ/min(self.n_histograms, self.total_batches)

        self.hist_num_of_bin = np.asarray(self.hist_num_of_bin).T
        
        return summ


