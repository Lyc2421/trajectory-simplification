from rl_env_inc import TrajComp
from rl_brain import PolicyGradient
import matplotlib.pyplot as plt
import time

def run_online(elist): # Validation
    eva = []
    total_len = []
    for episode in elist:
        #print('online episode', episode)
        total_len.append(len(env.ori_traj_set[episode]))
        buffer_size = int(ratio*len(env.ori_traj_set[episode]))
        if buffer_size < 3:
            continue
        steps, observation = env.reset(episode, buffer_size)
        for index in range(buffer_size, steps):
            if index == steps - 1:
                done = True
            else:
                done = False
            action = RL.pro_choose_action(observation)
            #action = RL.quick_time_action(observation) #use it when your model is ready for efficiency
            observation_, _ = env.step(episode, action, index, done, 'V') #'T' means Training, and 'V' means Validation
            observation = observation_
        eva.append(env.output(episode, 'V')) #'T' means Training, 'V' means Validation, and 'V-VIS' for visualization on Validation
    return eva
        
def run_comp(): #Training
    check = 999999
    training = []#放的是每50回合训练的误差，应该是用于可视化的，只是这个代码里最终没有使用到这个以及下面的验证列表
    validation = []#放的是每50回合验证的平均误差
    Round = 3
    while Round!=0:
        Round = Round - 1
        for episode in range(0, traj_amount):
            #print('training', episode)
            buffer_size = int(ratio*len(env.ori_traj_set[episode]))
            # extreme cases
            if buffer_size < 3:
                continue
            steps, observation = env.reset(episode, buffer_size)#初始化状态值，返回轨迹点数和有序列表前k个状态值
            for index in range(buffer_size, steps):#从第一次缓存外第一个点遍历到最后一个点
                #print('index', index)
                if index == steps - 1:#如果已经是轨迹最后一个点
                    done = True
                else:
                    done = False
                
                # RL choose action based on observation
                action = RL.pro_choose_action(observation)#状态输出到神经网络输出动作的概率分布，按概率采样一个动作
                #print('action', action)
                # RL take action and get next observation and reward
                observation_, reward = env.step(episode, action, index, done, 'T') #'T' means Training, and 'V' means Validation
                
                RL.store_transition(observation, action, reward)
                
                if done:
                    vt = RL.learn()
                    break
                # swap observation
                observation = observation_
            train_e = env.output(episode, 'T') #'T' means Training, 'V' means Validation, and 'V-VIS' for visualization on Validation
            show = 50
            if episode % show == 0:
                 eva = run_online([i for i in range(traj_amount, traj_amount + valid_amount - 1)])
                 #print('eva', eva)
                 res = sum(eva)/len(eva)
                 training.append(train_e)
                 validation.append(res)
                 print('Training error: {}, Validation error: {}'.format(sum(training[-show:])/len(training[-show:]), res))
                 RL.save('./save/'+ str(res) + '_ratio_' + str(ratio) + '/trained_model.ckpt')
                 print('Save model at round {} episode {} with error {}'.format(3 - Round, episode, res))
                 if res < check:
                     check = res
                 print('==>current best model is {} with ratio {}'.format(check, ratio))
    return training, validation

if __name__ == "__main__":
    # building subtrajectory env
    traj_path = '../TrajData/Geolife_out/'
    traj_amount = 1000
    valid_amount = 100
    a_size = 3
    s_size = 3
    ratio = 0.1 #缓存空间占轨迹大小的比例
    env = TrajComp(traj_path, traj_amount + valid_amount, a_size, s_size)
    RL = PolicyGradient(env.n_features, env.n_actions)
    #RL.load('./save/your_model/')
    start = time.time()
    training, validation = run_comp()
    print("Training elapsed time = %s", float(time.time() - start))
    plt.figure()
    plt.plot(training, "r",validation,"b")
    plt.show()
    