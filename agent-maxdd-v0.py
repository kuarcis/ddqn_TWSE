import env
import random, math
import os
import json
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
TAU = 0.1

BATCH = 50
MEM_CAP = 20000

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

    IsTriained = False
    game_count = 0
    total_step_cnt = 0
    set_reward = 0
    set_cnt = 0
    set_number = 0
    set_avg_r = []

    obs_space = ENV_SHAPE
    act_space = ACT_LEN

    avg_range = 100

    #init agent
    game_agent = agent(obs_space, act_space)
    print('agent inited')
    game_agent.brain.load_weight(game_agent.brain.model,verbose = 1)
    game_agent.brain.load_weight(game_agent.brain.t_model,verbose = 1)
    # logger = game_agent.brain.log()
    e= game_agent.brain.earlystop()
    alt_training = False

    try:
        print('training init')
        r_accmulate = 0
        critic_loss =0
        step_count = 0
        t_update_cnt = 0
        vaild_neg_cnt =0
        neg_cnt = 0
        nan_cnt = 0
        maxdd_cnt =0
        pack_s = game_env.observe()
        r_avg = []
        err_avg = []
        last_error = 0
        type_1_table = []   #type 1 error count table for all options
        type_2_table= []    #type 2 error count table for all options
        match_table= []     #match count table for all options
        for t in range(ACT_LEN):
            type_1_table.append([])
            type_2_table.append([])
            match_table.append([])
        # false_negative = [] # type 1 error
        # false_postive = [] # type 2 error
        # pred_acc =[[],[],[],[]]
        

        while True:
            
            if pack_s[2] or (not pack_s[2] and random.randint(0,1000)<300):
    
                # set every 1000 step as a "game set", for last guess of the set
                # s_ = None,
                # maybe it'll hepl converging error 0.0


                # print('\033[H\033[J')
                if pack_s[2]:
                    maxdd_cnt +=1
                
                # print(s_tmp)
                # print(stop)

                a = game_agent.act(np.array([pack_s[0]]))[0]
                a_tmp = a
                # print(a)
                #Ornstein-Uhlenbeck process
                a_tmp[0] += (game_agent.epsilon/1000)*((0.05-a_tmp[0]) +random.normalvariate(0.05,SIGMA))
                a_tmp[1] += (game_agent.epsilon/1000)*((0.05-a_tmp[1]) +random.normalvariate(0.05,SIGMA))
                a_tmp[2] += (game_agent.epsilon/1000)*((0.05-a_tmp[2]) +random.normalvariate(0.05,SIGMA))
                a_tmp[3] += (game_agent.epsilon/1000)*((0.05-a_tmp[3]) +random.normalvariate(0.05,SIGMA))
                a_tmp[4] += (game_agent.epsilon/1000)*((0.05-a_tmp[4]) +random.normalvariate(0.05,SIGMA))

                
                a =a_tmp
                # print(a)
                pred_a = np.argmax(a)
                # action space:
                # 0 : no maxdd reach
                # 1 : maxdd happened in 15 days
                # 2 : maxdd happened in 16-30 days
                # 3 : maxdd happened in 31-45 days
                # 4 : maxdd happened in 46-90 days

                #convert answer provided by env to options
                if not pack_s[2]:
                    answer = 0
                else:
                    answer = int(pack_s[1]/(TAR_RANGE/(ACT_LEN-1)))+1

                if answer != pred_a:
                    for i in range(ACT_LEN):
                        if i == answer: 
                            type_1_table[i].append(1)
                            type_2_table[i].append(0)
                            match_table[i].append(0)
                        elif i == pred_a:
                            type_1_table[i].append(0)
                            type_2_table[i].append(1)
                            match_table[i].append(0)

                else:
                    for i in range(ACT_LEN):
                        if i == answer:                            
                            type_1_table[i].append(0)
                            type_2_table[i].append(0)
                            match_table[i].append(1)


                # if not pack_s[2]:
                #     if pred_a != 0:
                #         false_postive.append(1)
                #     else:
                #         false_postive.append(0)
                # else:                    
                #     if pred_a != 0:                        
                #         false_negative.append(0)
                #         #prediction is select the right range
                #         if (pred_a-1) *30 < pack_s[1] and pred_a * 30 >= pack_s[1]:
                #             pred_acc[pred_a-1].append(1)
                #         else:
                #             # rewarding error range, it's important that this AI can tell maxdd happened
                #             # than it can predict the maxdd date precisely
                #             pred_acc[pred_a-1].append(0)
                #     else:
                #         # false negative define as : maxdd is reach, but AI says there's no maxdd happened
                        # false_negative.append(1)

                r = game_env.eval(a,pack_s)

                # from the idea of alphago training
                # assuming every 1000 step is a game set, only the last round will record a net reward
                # for the other rounds , reward = 0
                set_reward += r
                if step_count % 100 == 99:
                    
                    game_agent.observe( ( np.array(pack_s[0]), pred_a, r  , None))
                    pack_s_ = game_env.observe()
                    print('set',set_number,'done, set reward:',set_reward)
                    set_number += 1
                    set_avg_r.append(set_reward)
                    set_reward =0
                    
                    if set_number % 10 ==9:

                        # add training inprovement test
                        # if inprovement happened, continue training
                        train_cnt = 0
                        loss_tmp = 0
                        min_loss_try_cnt =0
                        
                        
                        print('training....')
                        game_agent.brain.save_weight(game_agent.brain.model)
                        if alt_training == False:
                            if sum(set_avg_r)/len(set_avg_r) < 100: 
                            #different training strategy before and after 100000 time of training
                                train_history = game_agent.replay(callback=[e])
                                game_agent.brain.t_update(game_agent.brain.model,game_agent.brain.t_model)
                                t_update_cnt +=1
                            else:
                                alt_training = True

                        else:
                            print('selecting higher score...')
                            total_val_r = 0
                            for i  in range(1000):
                                val_s = game_env.observe()
                                val_a = game_agent.act(np.array([val_s[0]]),israndom=False)[0]
                                val_r = game_env.eval(val_a,val_s)
                                total_val_r += val_r
                            
                            while min_loss_try_cnt < MIN_LOSS_IMPROVE_RUN:
                                train_cnt +=1
                                if train_cnt >10000:
                                    print('pass 10000 train, break')
                                    break
                                total_val_r_training = 0
                                train_history = game_agent.replay(callback=[e]) 
                                for i  in range(1000):
                                    val_s = game_env.observe()
                                    val_a = game_agent.act(np.array([val_s[0]]),israndom=False)[0]
                                    val_r = game_env.eval(val_a,val_s)
                                    total_val_r_training += val_r
                                
                                if total_val_r_training <total_val_r:                                
                                    # reload best outcome when loss not improve
                                    game_agent.brain.load_weight(game_agent.brain.model,verbose =0)
                                    min_loss_try_cnt +=1

                                else:                                
                                    game_agent.brain.t_update(game_agent.brain.model,game_agent.brain.t_model)
                                    t_update_cnt +=1
                                    game_agent.brain.save_weight(game_agent.brain.model)
                                    total_val_r = total_val_r_training                            
                                    min_loss_try_cnt =0
                            print('val_total_return:',total_val_r )

                        val_loss= train_history.history['val_loss']                
                        critic_loss = val_loss[len(val_loss)-1]    

                        
                        # keep last train data
                        
                        err_avg.append(critic_loss)
                        IsTriained = True

                else:
                    pack_s_ = game_env.observe()
                    game_agent.observe( ( np.array(pack_s[0]), pred_a, r, np.array(pack_s_[0])))
                


                # if len(false_negative) >1000:
                #     false_negative.pop(0)
                
                # if len(false_postive) >1000:
                #     false_postive.pop(0)
                for i in range(ACT_LEN):
                    if len(type_1_table[i])>1000:
                        type_1_table[i].pop(0)
                    if len(type_2_table[i])>1000:
                        type_2_table[i].pop(0)
                    if len(match_table[i])>1000:
                        match_table[i].pop(0)

                if len(err_avg) >avg_range:
                    err_avg.pop(0)

                if len(set_avg_r)>100:
                    set_avg_r.pop(0)


                # for selection_acc in range(len(pred_acc)):
                #     if len(pred_acc[selection_acc])>1000:
                #         pred_acc[selection_acc].pop(0)

                step_count +=1

                if step_count % 1800 == 0:
                    print('step count:',step_count,',set count:',set_number)
                    if len(set_avg_r) != 0:
                        print('average set reward in recent 100 set:',format(sum(set_avg_r)/len(set_avg_r),'.2f'))
                    if IsTriained:
                        print('avg critic val_err form recent 100 train:', format(math.fsum(err_avg)/len(err_avg),'.5f'))
                    print('memory length: ',game_agent.memory.len(),', maxdd_cnt:',maxdd_cnt,',target netwrok update time:', t_update_cnt)
                    for i in range(ACT_LEN):
                        if len(type_1_table[i]) >0 and len(type_2_table[i])>0 and len(match_table[i]) >0:
                            print('option:',i,', accuracy:',format(sum(match_table[i])/len(match_table[i]),'7.2%'),\
                            ' ,type 1 error:',format(sum(type_1_table[i])/len(type_1_table[i]),'7.2%'),\
                            ' ,type 2 error:',format(sum(type_2_table[i])/len(type_2_table[i]),'7.2%'))
                            
                    # if len(false_negative) !=0 and len(false_postive) != 0:
                    #     print('type 1 error ratio in recent 1000 round: ', format(sum(false_negative)/len(false_negative), '.2%') ,\
                    #     'type 2 error ratio in recent 1000 round: ', format(sum(false_postive)/len(false_postive), '.2%') ,  )
                    # for selection_acc in range(len(pred_acc)):
                    #     if len(pred_acc[selection_acc])>0:
                    #         print('prediction accuracy of selection' ,selection_acc+1,'in recent 1000 prediction::',format(sum(pred_acc[selection_acc])/len(pred_acc[selection_acc]),'.2%'))
                    #     else:
                    #         print('prediction accuracy of selection' ,selection_acc+1,': not enought sample')

                pack_s = pack_s_
                
            else:
                pack_s = game_env.observe()

    except KeyboardInterrupt:
        print('keyboard interrupted')
        pass
    finally:

        if IsTriained:
            print('stopped')
            game_agent.brain.save_weight(game_agent.brain.model,verbose = 1)

        else:
            print('no training occur')
        print('end training')
    
