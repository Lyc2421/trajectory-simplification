import numpy as np
import data_utils as F
#import heapq
import copy
#from heapq import heappush, heappop, _siftdown, _siftup
import matplotlib.pyplot as plt
import math
from sortedcontainers import SortedList
import random
import os
import time

class TrajComp():
    def __init__(self, a_size, s_size):
        self.n_actions = a_size
        self.n_features = s_size

    def set_error_type(self, label):
        self.label = label
        if label == 'sed':
            self.op = lambda segment: F.sed_op(segment)
            self.err_comp = lambda ori_traj, sim_traj: F.sed_error(ori_traj, sim_traj)
        elif label == 'ped':
            self.op = lambda segment: F.ped_op(segment)
            self.err_comp = lambda ori_traj, sim_traj: F.ped_error(ori_traj, sim_traj)
        elif label == 'dad':
            self.op = lambda segment: F.dad_op(segment)
            self.err_comp = lambda ori_traj, sim_traj: F.dad_error(ori_traj, sim_traj)
        elif label == 'sad':
            self.op = lambda segment: F.speed_op(segment)
            self.err_comp = lambda ori_traj, sim_traj: F.speed_error(ori_traj, sim_traj)
        else:
            print("Label is wrong.")

    def load_one_sample(self, path, index):
        self.ori_traj_set = []
        self.ori_traj_set.append(F.to_traj(path + str(index)))

    def load_train_data(self, path, traj_amount, valid_amount, cut=False):
        print("======loading train data======")
        start = time.time()
        self.traj_amount = traj_amount
        amount = traj_amount + valid_amount
        self.ori_traj_set = []
        if not cut:
            for num in range(amount):
                self.ori_traj_set.append(F.to_traj(path + str(num)))
        else:
            print('cut=',1000)
            num = 0
            while amount > 0:
                traj = F.to_traj(path + str(num))
                if len(traj) >= 1000:
                    self.ori_traj_set.append(traj[:1000])
                    amount -= 1
                num += 1
        print("It cost {}s.".format(float(time.time()-start)))

    def load_test_data(self, path, amount, cut=False):
        print("======loading test data======")
        start = time.time()
        self.ori_traj_set = []
        total_len = len(os.listdir(path))
        if not cut:
            for num in range(total_len - amount, total_len):
                self.ori_traj_set.append(F.to_traj(path + str(num)))
        else:
            num = total_len - 1
            while amount > 0:
                traj = F.to_traj(path + str(num))
                if len(traj) >= 1000:
                    self.ori_traj_set.append(traj[:1000])
                    amount -= 1
                num -= 1
        print("It cost {}s".format(float(time.time()-start)))

    def shuffle(self):
        traj_set = self.ori_traj_set[:self.traj_amount]
        random.shuffle(traj_set)
        self.ori_traj_set[:self.traj_amount] = traj_set

    def read(self, p, episode, rem, flag):
        self.F_ward[self.link_tail] = [0.0, p] #新加进来的点，初始化它在两个字典里的状态值和索引
        self.B_ward[p] = [0.0, self.link_tail] 
        s = self.B_ward[self.link_tail][1]#尾部点前一个点的索引
        m = self.link_tail#尾部点的索引
        e = self.F_ward[self.link_tail][1]#尾部点后一个点的索引，也就是新加进来的点的索引
        if flag:#rem是要删除的点
            self.F_ward[m][0] = self.op([self.ori_traj_set[episode][s], self.ori_traj_set[episode][rem], self.ori_traj_set[episode][m], self.ori_traj_set[episode][e]])
            self.B_ward[m][0] = self.F_ward[m][0]
        else:#计算出尾部点的状态值
            self.F_ward[m][0] = self.op([self.ori_traj_set[episode][s], self.ori_traj_set[episode][m], self.ori_traj_set[episode][e]])
            self.B_ward[m][0] = self.F_ward[m][0]
        #heapq.heappush(self.heap, (self.F_ward[m][0], m))# save (state_value, point index of ori traj)
        self.sortedlist.add((self.F_ward[m][0], m))#把尾部点的状态值和索引组成的元组插入排序列表中，会自动维持有序状态
        self.link_tail = p#把尾部索引改成新加进来的点的索引
    
    def reset(self, episode, buffer_size):#初始状态值是轨迹中前缓存大小个点（除了第一个）的状态值，相对于左右相邻点组成的锚段的误差值。
        #self.heap = []
        self.last_error = 0.0
        self.current = 0.0
        self.c_left = 0
        self.c_right = 0
        #self.copy_traj = copy.deepcopy(self.ori_traj_set[episode]) #for testing the correctness of inc rewards
        self.start = {}
        self.end = {}
        self.err_seg = {}
        steps = len(self.ori_traj_set[episode])#步骤就是轨迹的点数
        self.F_ward = {} # save (state_value, next_point)保存的是简化轨迹的索引下个点的信息的字典
        self.B_ward = {} # save (state_value, last_point)保存的是简化轨迹的索引上个点的信息的字典
        self.F_ward[0] = [0.0, 1]
        self.B_ward[1] = [0.0, 0]
        self.link_head = 0
        self.link_tail = 1
        self.sortedlist = SortedList({})
        for i in range(2, buffer_size + 1):#缓存大小是轨迹长度的十分之一
            self.read(i, episode, -1, False)
        self.check = [self.sortedlist[0][1], self.sortedlist[0][1], self.sortedlist[1][1]]#状态值的点的索引
        self.state = [self.sortedlist[0][0], self.sortedlist[0][0], self.sortedlist[1][0]]#状态值，有序列表前三个，初始列表中可能只有2个元素（buffer是3的时候） 
        #print('len, obs, heap and state', len(self.heap), self.observation, self.heap, self.state)
        return steps, np.array(self.state).reshape(1, -1)#数组变成[[1,2,3]]形式，形状是（1,3）           
        
    def reward_update(self, episode, rem):
        if (rem not in self.start) and (rem not in self.end):
            #interval insert
            a = self.B_ward[rem][1]#被删点上一个点
            b = self.F_ward[rem][1]#被删点下一个点
            self.start[a] = b
            self.end[b] = a
            NOW = self.op(self.ori_traj_set[episode][a: b + 1])
            self.err_seg[(a,b)] = NOW
            if NOW >= self.last_error:
                self.current = NOW
                self.current_left, self.current_right = a, b
        
        elif (rem in self.start) and (rem not in self.end):
            #interval expand left
            a = self.B_ward[rem][1]
            b = rem
            c = self.start[rem]
            BEFORE = self.err_seg[(b,c)]
            NOW = self.op(self.ori_traj_set[episode][a: c + 1])
            del self.err_seg[(b,c)]
            self.err_seg[(a,c)] = NOW
            
            if  math.isclose(self.last_error,BEFORE):
                if NOW >= BEFORE:
                    #interval expand left_case1
                    self.current = NOW
                    self.current_left, self.current_right = a, c
                else:
                    #interval expand left_case2
                    (self.current_left, self.current_right) = max(self.err_seg, key=self.err_seg.get)
                    self.current = self.err_seg[(self.current_left, self.current_right)]
            else:
                #interval expand left_case3
                if NOW >= self.last_error:
                    self.current = NOW
                    self.current_left, self.current_right = a, c
            self.end[c] = a
            self.start[a] = c
            del self.start[b]
            
        # interval expand right
        elif (rem not in self.start) and (rem in self.end):
            #interval expand right
            a = self.end[rem]
            b = rem
            c = self.F_ward[rem][1]
            BEFORE = self.err_seg[(a,b)]
            NOW = self.op(self.ori_traj_set[episode][a: c + 1])
            del self.err_seg[(a,b)]
            self.err_seg[(a,c)] = NOW
            if math.isclose(self.last_error,BEFORE):
                if NOW >= BEFORE:
                    #interval expand right_case1
                    self.current = NOW
                    self.current_left, self.current_right = a, c
                else:
                    #interval expand right_case2
                    (self.current_left, self.current_right) = max(self.err_seg, key=self.err_seg.get)
                    self.current = self.err_seg[(self.current_left, self.current_right)]
            else:
                #interval expand right_case3
                if NOW >= self.last_error:
                    self.current = NOW
                    self.current_left, self.current_right = a, c
            self.start[a] = c
            self.end[c] = a
            del self.end[b]
        
        # interval merge
        elif (rem in self.start) and (rem in self.end):
            #interval merge
            b = rem
            a = self.end[b]
            c = self.start[b]
            # get values quickly
            BEFORE_1 = self.err_seg[(a,b)]
            BEFORE_2 = self.err_seg[(b,c)]
            NOW = self.op(self.ori_traj_set[episode][a: c + 1])
            del self.err_seg[(a,b)]
            del self.err_seg[(b,c)]
            self.err_seg[(a,c)] = NOW            
            if math.isclose(self.last_error,BEFORE_1):
                if NOW >= BEFORE_1:
                    #interval merge_case1
                    self.current = NOW
                    self.current_left, self.current_right = a, c
                else:
                    #interval merge_case2
                    (self.current_left, self.current_right) = max(self.err_seg, key=self.err_seg.get)
                    self.current = self.err_seg[(self.current_left, self.current_right)]
                    
            elif math.isclose(self.last_error,BEFORE_2):
                if NOW >= BEFORE_2:
                    #interval merge_case3
                    self.current = NOW
                    self.current_left, self.current_right = a, c
                else:
                    #interval merge_case4
                    (self.current_left, self.current_right) = max(self.err_seg, key=self.err_seg.get)
                    self.current = self.err_seg[(self.current_left, self.current_right)]
            else:
                #interval merge_case5
                if NOW >= self.last_error:
                    self.current = NOW
                    self.current_left, self.current_right = a, c
                    
            self.start[a] = c
            self.end[c] = a
            del self.start[b]
            del self.end[b]
        else:
            print('Here is a bug!!!')
    
    def delete_heap(self, heap, nodeValue):
        leafValue = heap[-1]
        i = heap.index(nodeValue)
        if nodeValue == leafValue:
            heap.pop(-1)
        elif nodeValue <= leafValue: # similar to heappop
            heap[i], heap[-1] = heap[-1], heap[i]
            minimumValue = heap.pop(-1)
            if heap != []:
                _siftup(heap, i)
        else: # similar to heappush
            heap[i], heap[-1] = heap[-1], heap[i]
            minimumValue = heap.pop(-1)
            _siftdown(heap, 0, i)
        
    def step(self, episode, action, index, done, label = 'T'):        
        # update state and compute reward更新状态和计算奖励
        #标签的T是训练，V是验证，index是这一步新添加进来的点的索引，done是标志新添加进来的点是不是轨迹最后一个点

        rem = self.check[action] # point index in ori traj动作要删除的点的索引

        NEXT_P = self.F_ward[rem][1]#要删除的点的下一个点的索引
        NEXT_V = self.B_ward[NEXT_P][0]
        LAST_P = self.B_ward[rem][1]#要删除的点的上一个点的索引
        LAST_V = self.F_ward[LAST_P][0]

        if LAST_P > self.link_head:#如果上一个点不是第一个点
            #self.delete_heap(self.heap, (LAST_V, LAST_P))
            self.sortedlist.remove((LAST_V, LAST_P))#有序列表中删除上一个点的元组
            s = self.ori_traj_set[episode][self.B_ward[LAST_P][1]]
            m1 = self.ori_traj_set[episode][LAST_P]
            m2 = self.ori_traj_set[episode][rem]
            e = self.ori_traj_set[episode][NEXT_P]
            self.F_ward[LAST_P][0] = self.op([s,m1,m2,e])#?
            self.B_ward[LAST_P][0] = self.F_ward[LAST_P][0]
            #heapq.heappush(self.heap, (self.F_ward[LAST_P][0], LAST_P))
            self.sortedlist.add((self.F_ward[LAST_P][0], LAST_P))#重新算删除点前一个点的状态值然后插入有序列表
        if NEXT_P < self.link_tail:#如果下一个点不是最后一个点
            #self.delete_heap(self.heap, (NEXT_V, NEXT_P))
            self.sortedlist.remove((NEXT_V, NEXT_P))#有序列表中删除下一个点的元组
            s = self.ori_traj_set[episode][LAST_P]
            m1 = self.ori_traj_set[episode][rem]
            m2 = self.ori_traj_set[episode][NEXT_P]
            e = self.ori_traj_set[episode][self.F_ward[NEXT_P][1]]
            self.F_ward[NEXT_P][0] = self.op([s,m1,m2,e])#?
            self.B_ward[NEXT_P][0] = self.F_ward[NEXT_P][0]
            #heapq.heappush(self.heap, (self.F_ward[NEXT_P][0], NEXT_P))
            self.sortedlist.add((self.F_ward[NEXT_P][0], NEXT_P))#重新算删除点后一个点的状态值然后插入有序列表
            
        #self.copy_traj.remove(self.ori_traj_set[episode][rem]) #for testing the correctness of inc rewards
        if  label == 'T':#如果是训练状态就更新奖励
            self.reward_update(episode, rem)
        
        self.F_ward[LAST_P][1] = NEXT_P
        self.B_ward[NEXT_P][1] = LAST_P
        #self.delete_heap(self.heap, (self.F_ward[rem][0], rem))
        self.sortedlist.remove((self.F_ward[rem][0], rem))
        del self.F_ward[rem]#在前向和后向字典里都删去这个点
        del self.B_ward[rem]     
        
        #_,  self.current = F.sed_error(self.ori_traj_set[episode], self.copy_traj) #for testing the correctness of inc rewards
        rw = self.last_error - self.current#奖励就是这一步简化前的误差-简化后的误差（都是相对于原始轨迹的误差）
        self.last_error = self.current
        #print('self.current',self.current)            
            
#        if not done: #boundary process
#            if NEXT_P == self.link_tail:
#                self.read(index + 1, episode, rem, True)
#                self.check = [self.heap[0][1], LAST_P, LAST_P]
#                self.state = [self.heap[0][0], self.F_ward[LAST_P][0], self.F_ward[LAST_P][0]]
#            else:
#                self.read(index + 1, episode, rem, False)
#                if LAST_P == self.link_head:
#                    self.check = [self.heap[0][1], NEXT_P, NEXT_P]
#                    self.state = [self.heap[0][0], self.B_ward[NEXT_P][0], self.B_ward[NEXT_P][0]]
#                else:
#                    self.check = [self.heap[0][1], LAST_P, NEXT_P]
#                    self.state = [self.heap[0][0], self.F_ward[LAST_P][0], self.B_ward[NEXT_P][0]]
        
        if not done: #boundary process
            if NEXT_P == self.link_tail:
                self.read(index + 1, episode, rem, True)
                if len(self.sortedlist) < self.n_features: #仅仅是应对W和k相等的情况，第一个点无法删，只有k-1个选择。
                    self.check = [self.sortedlist[0][1],self.sortedlist[0][1], self.sortedlist[1][1]]
                    self.state = [self.sortedlist[0][0],self.sortedlist[0][0], self.sortedlist[1][0]]
                else:
                    t = self.sortedlist[:self.n_features]
                    self.check = [t[0][1], t[1][1], t[2][1]]
                    self.state = [t[0][0], t[1][0], t[2][0]]
                    
            else:
                self.read(index + 1, episode, rem, False)
                if len(self.sortedlist) < self.n_features:
                    self.check = [self.sortedlist[0][1],self.sortedlist[0][1],self.sortedlist[1][1]]
                    self.state = [self.sortedlist[0][0],self.sortedlist[0][0],self.sortedlist[1][0]]
                else:
                    t = self.sortedlist[:self.n_features]
                    self.check = [t[0][1], t[1][1], t[2][1]]
                    self.state = [t[0][0], t[1][0], t[2][0]]
        
        #print('heap', self.heap)
        #print('check and state', self.check, self.state)
        return np.array(self.state).reshape(1, -1), rw#返回每一步行动后更新的状态，以及奖励
    
    def output(self, episode, label = 'T'):#返回的是整条简化后的轨迹和原始轨迹的误差，以及可视化
        if label == 'V-VIS':#验证并可视化
            start = 0
            sim_traj = []
            while start in self.F_ward:
                sim_traj.append(self.ori_traj_set[episode][start])
                start = self.F_ward[start][1]
            sim_traj.append(self.ori_traj_set[episode][start])
            _, final_error = self.err_comp(self.ori_traj_set[episode], sim_traj)
            # print('Validation at episode {} with error {}'.format(episode, final_error))
            #for visualization, 'sed' is by default, if you want to draw other errors by revising the codes in data_utils.py correspondingly.
            F.draw(self.ori_traj_set[episode], sim_traj, label=self.label) 
            return final_error
        if label == 'V':#验证
            start = 0
            sim_traj = []
            while start in self.F_ward:
                sim_traj.append(self.ori_traj_set[episode][start])
                start = self.F_ward[start][1]
            sim_traj.append(self.ori_traj_set[episode][start])
            _, final_error = self.err_comp(self.ori_traj_set[episode], sim_traj)
            return final_error
        if label == 'T':#训练
            # print('Training at episode {} with error {}'.format(episode, self.current))
            return self.current
