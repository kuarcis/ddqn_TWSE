import env
import random, math
import os,csv
import json
import datetime
from time import localtime, strftime
import numpy as np

from keras.models import *
from keras.layers import *
from keras.preprocessing.sequence import *
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import *
from keras.callbacks import *
from keras.regularizers import *
from keras.layers.normalization import *
from statistics import *
import keras.backend as K
import tensorflow as tf



HUBER_LOSS_DELTA = 5.0
#=============hubert loss ========
def huber_loss(y_true, y_pred):
    err = y_true - y_pred

    cond = K.abs(err) < HUBER_LOSS_DELTA
    L2 = 0.5 * K.square(err)
    L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)

    loss = tf.where(cond, L2, L1)   # Keras does not cover where function in tensorflow :-(

    return K.mean(loss)
#================================



GAME_NAME= 'maxdd'

OBS_RANGE = 250     # 2 year
TAR_RANGE = 120     # 120 days
MAX_DD_LEN = 10     # in 10 days
MAX_DD = 0.07       # drop 7%

# env shape  for the TWSE_M_institution file:8
# other file is indicated with factor number
ENV_SHAPE = 10
ACT_LEN= 5         # [no maxdd reach, reach in 30 days, reach in 30-60 days, 
                    # reach in 60-90 days, reach in 90-120 days]
                    # action sapce = 5
TAU = 0.002

BATCH = 50
MEM_CAP = 2000

MIN_LOSS_IMPROVE_RUN = 10

SIGMA = 0.01
GAMMA = 0.1
MAX_EPSILON = 0.55
MIN_EPSILON = 0.01
LAMBDA = 1e-5      # speed of frequency decay
VER = "v0"
logging_dir = "./logs/"+VER+"/"+strftime("%Y%m%d%H%M%S", localtime())

# DDQN inplentation:
# use target network to cal Q value
# use main network to provide prediction(action)

class Brain:
    def __init__(self, state_space_shape, action_space_shape):

        self.state_space_shape = state_space_shape
        self.action_space_shape = action_space_shape
        self.tau = TAU

        self.model = self._VallinaNNModel()
        self.t_model = self._VallinaNNModel()

        print('model init')

    def _VallinaNNModel(self):
        # action output
        # a: [0,1], output as the probability of maxdd happened


        print('init dqn model')
        
        
        input_layer= Input(shape=(OBS_RANGE,self.state_space_shape,) )
        # LSTM_1 = LSTM(30)(input_layer)
        flat = Flatten()(input_layer)
        # layer1= Dense(603, activation='linear')(LSTM_1)
        # layer1= Dense(OBS_RANGE, activation='linear')(flat)
        layer1= Dense(200, activation='linear')(flat)
        layer1= BatchNormalization()(layer1)
        layer1 = Dropout(0.5)(layer1)
 
        layer2= Dense(100, activation='linear')(layer1)
        layer2 =LeakyReLU(alpha=0.3)(layer2)
        layer2 = Dropout(0.5)(layer2)

        sepe_layer1_1 = Dense(61, activation='linear')(layer2)
        sepe_layer1_1 =LeakyReLU(alpha=0.3)(sepe_layer1_1)
        sepe_layer1_1 = Dropout(0.5)(sepe_layer1_1)

        sepe_layer1_2 = Dense(81, activation='linear')(layer2)
        sepe_layer1_2 =LeakyReLU(alpha=0.3)(sepe_layer1_2)
        sepe_layer1_2 = Dropout(0.5)(sepe_layer1_2)

        sepe_layer2_1 = Dense(51, activation='linear')(sepe_layer1_1)
        sepe_layer2_1 =LeakyReLU(alpha=0.3)(sepe_layer2_1)
        sepe_layer2_1 = Dropout(0.5)(sepe_layer2_1)

        sepe_layer2_2 = Dense(71, activation='linear')(sepe_layer1_2)
        sepe_layer2_2 =LeakyReLU(alpha=0.3)(sepe_layer2_2)
        sepe_layer2_2 = Dropout(0.5)(sepe_layer2_2)

        # seperate no maxdd preditction(array 0) and other option
        out_1 = Dense(1, activation='linear',kernel_regularizer=l2(1e-5))(sepe_layer2_1)
        out_2 = Dense(self.action_space_shape-1 ,activation = 'linear',kernel_regularizer=l2(1e-5))(sepe_layer2_2 )
        
        out = concatenate([out_1,out_2])

        # opt = RMSprop(lr=0.001)
        opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
        model = Model(input_layer, out)

        model.compile(optimizer = opt, loss= 'mse')
        print('dqn model inited')        
        return model

    
    def save_weight(self,model,tmp_target_save = False, filename=GAME_NAME + "-"+ VER, verbose = 0):

        if verbose == 1:
            print("\nsave weight")
        if tmp_target_save:
            model.save_weights('tmp.h5', overwrite=True)
            with open('tmp.json','w') as tmp_out:
                json.dump(model.to_json(),tmp_out)
        else:
            model.save_weights(filename+"_dqn.h5", overwrite=True)
            with open(filename+'_dqn.json', "w") as outfile:
                json.dump(model.to_json(), outfile)
        if verbose == 1:
            print('dqn model save')


    def load_weight(self,model, filename=GAME_NAME+"-"+ VER, verbose = 0 ):
        t_model = model
        try:
            t_model.load_weights(filename+"_dqn.h5")
            if verbose == 1:
                print("Weight load successfully from " + filename+ "_dqn.h5, continue")
        except:
            if verbose == 1:
                print("Cannot find the weight "+ filename+"_dqn.h5, continue")
 

    # def log(self):
        
    #     logger = TensorBoard(log_dir=logging_dir, histogram_freq=1, batch_size=32, write_graph=True,\
    #     write_grads=True)

    #     return logger

    def earlystop(self):
        
        e = EarlyStopping(monitor='loss',min_delta=1e-5, patience=5)

        return e

    def train(self, x, y, epoch=10, verbose=0, validation_split=0.1, train_Callback=None):
   
        train_hist = self.model.fit(x, y, batch_size=BATCH, epochs=epoch, validation_split=validation_split,\
         verbose=verbose, callbacks=train_Callback)
        return train_hist

    def predict(self, s):
        # print(s)
        return self.model.predict(s)
    
    # target network update
    # put TAU portion of main_nn weight to tar_nn
    # input must be model itself
    def t_update(self , main_nn, tar_nn):
        main_nn_weights = main_nn.get_weights()
        tar_nn_weigths = tar_nn.get_weights()
        for i in range(len(main_nn_weights)):
            tar_nn_weigths[i] = self.tau* main_nn_weights[i]+ (1-self.tau)*tar_nn_weigths[i]
        tar_nn.set_weights(tar_nn_weigths)


class Memory:   # stored as ( s, a, r )
    samples = []
    def __init__(self, capacity=MEM_CAP):
        self.capacity = capacity

    def add(self, sample):
        self.samples.append(sample)        

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

    def len(self):
        return len(self.samples)


class agent:
    steps = 0
    epsilon = MAX_EPSILON
    
 
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt
        self.brain = Brain(stateCnt, actionCnt)
        self.memory = Memory()

    def act(self, s,israndom = True):
        if random.random() < self.epsilon:
            rnd_act0 = random.uniform(-5,2)
            rnd_act1 = random.uniform(-5,2)
            rnd_act2 = random.uniform(-5,2)
            rnd_act3 = random.uniform(-5,2)
            rnd_act4 = random.uniform(-5,2)

            if israndom  == True:
                return [[rnd_act0,rnd_act1,rnd_act2,rnd_act3,rnd_act4]]
            else:
                return self.brain.predict(s)
        else:            
            return self.brain.predict(s)

    
    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)        

        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def replay(self, callback=None,eval = False):

        batch = self.memory.sample(BATCH*5)
        batchLen = len(batch)

        no_state = np.zeros((OBS_RANGE,self.stateCnt))
        # print(no_state.shape)


        states = np.array([ o[0] for o in batch ])
        states_ = np.array([ no_state if o[3] is None else o[3] for o in batch ])
 


        # try:
        
        q = self.brain.predict(states)
        

        q_ = self.brain.predict(states_)
        target_q_ = self.brain.t_model.predict(states_)
        # except:
        #     # print(err)
        

        x = []
        y = []
        
 
        for i in range(batchLen):
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
  
            t = q[i]
            # print('t[a]:',t[a])
            # print('np.amax(q_[i]:',np.amax(q_[i]))
            if s_ is None:
                t[a] = r
            else:                
                t[a] = r + GAMMA *target_q_[i][ np.argmax(q_[i])]


            x.append(s)
            y.append(t)
        x = np.array(x)
        y= np.array(y)
        if eval is False:
            if callback is None:
                hist = self.brain.train(x, y)
            else:        
                hist = self.brain.train(x, y,train_Callback=callback)
        else:
            hist = self.brain.model.evaluate(x,y,batch_size=BATCH)
        return hist
        #updated critic network
        

if __name__ == '__main__':
    #====================call all function and class==============
    print('init maxdd prediction env')
    game_env = env.TWSE(obs_range =OBS_RANGE,tar_range =TAR_RANGE,max_dd=MAX_DD,max_dd_len=MAX_DD_LEN )
    
    print('init maxdd prediction env done')
   
    obs_space = ENV_SHAPE
    act_space = ACT_LEN



    #init agent
    game_agent = agent(obs_space, act_space)
    print('agent inited')
    game_agent.brain.load_weight(game_agent.brain.model,verbose = 1)
    game_agent.brain.load_weight(game_agent.brain.t_model,verbose = 1)
    # logger = game_agent.brain.log()
    e= game_agent.brain.earlystop()
    # print(game_env.maxdd_table)
    print('\033[H\033[J')
    try:
        sample_pool= copy.copy(game_env.stk_data) 
        time_range = sample_pool.index.tolist()

        vaild_sample_date = False
        print('\n')
        print('maxdd_dqn:ai predict if there is any fall of TWSE index ')
        print('that is larger than 7 percent in 10 days from the target date')
        print("till 120 days after the target date")
        
        print('\nraw output woudld bo one of these 5 options:')
        print('\n1:there is no such sudden fall in the next 120 days since target date')
        print('2:there will be a sudden fall in the range of 0-30 days after the target date')
        print('3:there will be a sudden fall in the range of 31-60 days after the target date')
        print('4:there will be a sudden fall in the range of 61-90 days after the target date')
        print('5:there will be a sudden fall in the range of 91-120 days after the target date')
		
        # batch output all prediction raw data
        print('batch output prediction...')
        all_pred_list = []
        for t in range(len(time_range)):
            if t >= OBS_RANGE:
                s = copy.copy( sample_pool.iloc[t-OBS_RANGE:t])
                s = s.as_matrix()
                s = s.tolist()
                val_a = game_agent.act(np.array([s]),israndom=False)[0]
                sel_a = np.argmax(val_a)
                all_pred_list.append([time_range[t],sel_a])
        #print(all_pred_list)
        with open("./batch_pred.csv", "r+") as outfile:
            wr= csv.writer(outfile, quoting=csv.QUOTE_ALL)
            wr.writerow(all_pred_list)
			
        print('\nend the sampler by ctrl+c')
        while True:
            vaild_sample_date = False
            while vaild_sample_date != True:
                print('\nplease enter a target date for ai to predict')
                print("input format:yyyy-mm-dd")
                sample_date_str = input("range:{0} ~ {1}\n".format(time_range[OBS_RANGE].date(),time_range[len(time_range)-1].date()))
                #format check
                try:
                    if sample_date_str[4] != '-' or sample_date_str[7] != '-':
                        raise Exception('wrong type of input, try again')
                    # print('1')
                    sample_date = datetime.datetime(int(sample_date_str[0:4]),int(sample_date_str[5:7]),int(sample_date_str[8:10]))
                    # print('2')
                    if sample_date < time_range[OBS_RANGE] or sample_date > time_range[len(time_range)-1]:
                        # print(sample_date - time_range[120])
                        # print(sample_date - time_range[len(time_range)-1])
                        raise Exception('input is out from data range, try again')
                    date_exist = False
                    t_possible = 0
                    seq_cnt =0
                    for t in time_range[OBS_RANGE:]:
                        
                        delta = t - sample_date
                        delta = delta.days
                        if delta >= -2 and delta <0:
                            t_possible = t
                        if t == sample_date:
                            date_exist = True                    
                            break
                        seq_cnt +=1
                    if date_exist == False:
                        raise Exception('target date is not inside data pool, try {0}'.format(t_possible.date()))
                    vaild_sample_date = True
                except Exception as e:
                    print(e)
                # except err
            print(sample_date.strftime("%Y-%m-%d"),'selected, sample seq:',seq_cnt)
            seq_cnt = seq_cnt +OBS_RANGE
            s = copy.copy( sample_pool.iloc[seq_cnt-OBS_RANGE:seq_cnt])
            s = s.as_matrix()
            s = s.tolist()
            val_a = game_agent.act(np.array([s]),israndom=False)[0]
            print('\nraw output:',val_a)
            ai_selection = np.argmax(val_a)
            ai_selection =int(ai_selection)
            print('AI predict:',ai_selection)
            if ai_selection !=0:
                print('predict date range:\n',sample_date+datetime.timedelta((ai_selection-1)*30),'\~\n',sample_date+datetime.timedelta((ai_selection)*30))
            else:
                print('AI predict: no sudden drop will happened in the next',TAR_RANGE,' days')
            if seq_cnt < game_env.data_len - TAR_RANGE:
                print('precalculated answer\n(in the form of (drop_date_since_target_date, if_drop_happened):')
                print(game_env.maxdd_table[seq_cnt])


    except KeyboardInterrupt:
        print('keyboard interrupted')
        pass
    finally:

        print('end program')
    
