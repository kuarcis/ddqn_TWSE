import loader
import pandas
import random
import math
import copy

import numpy as np
import multiprocessing as mp

# modified for TWSE_M_institution file


class TWSE:
    obs_date = 0
    isReset = False

    def __init__(self, obs_range, tar_range, max_dd, max_dd_len):
        self.stk_data, self.ca_data, self.have_ca = loader.xls_loader()
        self.data_len = self.stk_data.shape[0] -1
        self.obs_range = obs_range
        self.max_dd = max_dd
        self.max_dd_len = max_dd_len
        self.tar_range = tar_range
        print('calculating maxdd situation....')
        self.worker()
        print('calculation done')

    def maxdd_check(self,seq):
        t_max = 0
        t_min = 0
        t_max_date = 0
        t_min_date = 0
        maxdd_reach = False
        # if seq% 100 == 0:
        #     print(seq, '/',len(self.stk_data))
        for j in range(self.tar_range-self.max_dd_len-1):
            t_max = self.stk_data.iloc[seq+j]['Close']
            t_max_date = j
            for  k in range(self.max_dd_len):
                t_min = self.stk_data.iloc[seq+j+k+1]['Close']
                if (t_min-t_max)/t_max < -self.max_dd:
                    maxdd_reach = True
                    return [seq,t_max_date,maxdd_reach]
                        
        return [seq,None,maxdd_reach]

 
    def worker(self):
        maxdd_table = []
        for i in range(self.data_len-self.tar_range):
            maxdd_table.append([])
        pool = mp.Pool()
        arg_set = []
        r = pool.map(self.maxdd_check,range(self.data_len-self.tar_range))
        pool.close()
        pool.join()
        for i in r:
            maxdd_table[i[0]] = [i[1],i[2]]
        print('maxdd table done. len:', len(maxdd_table))
        print('obs_range:',self.obs_range,',tar_range:',self.tar_range,',data_len',self.data_len)
        self.maxdd_table = maxdd_table 
          
                    

    def observe(self,seq = None):
        #only return s and if there is actual maxdd happened
        s = None
        self.obs_date = random.randrange(self.obs_range+1, self.data_len-self.tar_range-1)
        s =copy.copy( self.stk_data.iloc[self.obs_date-self.obs_range:self.obs_date])
        
        s = s.as_matrix()
        s = s.tolist()
        return [s, self.maxdd_table[self.obs_date][0], self.maxdd_table[self.obs_date][1]]
    def eval(self,a,maxdd_result):
        r = 0
        pred_a = np.argmax(a)
        if not maxdd_result[2]:
            if pred_a != 0:
                r = -4
            else:
                r = 1
        else:
            if pred_a != 0:
                #prediction is select the right range
                if (pred_a-1) *30 <maxdd_result[1] and pred_a * 30 >= maxdd_result[1]:
                    r = 2
                else:
                    # rewarding error range, it's important that this AI can tell maxdd happened
                    # than it can predict the maxdd date precisely
                    r=-1
            else:
                # false negative define as : maxdd is reach, but AI says there's no maxdd happened
                r = -5
        return r