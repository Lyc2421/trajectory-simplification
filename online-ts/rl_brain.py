import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
#import numba as nb

#for reproducible
np.random.seed(1)
tf.compat.v1.set_random_seed(1)

class PolicyGradient:
    def __init__(self, n_features, n_actions, learning_rate=0.001, reward_decay=0.99, output_graph=False):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.reward_baseline = 0
        self.alpha = 0.01
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        tf.compat.v1.reset_default_graph()
        self._build_net()
        self.sess = tf.Session()
        if output_graph:#把计算图和其他指标写入日志文件，用tensorboard可视化
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        #初始化所有全局变量，使得所有模型参数都有了一个初始值。sess.run执行这个
        #初始化，这一步正在TensorFlow会话中进行。

    def _build_net(self):
        with tf.name_scope('inputs'):#定义一个命名空间的上下文管理器，提供一个
            #前缀，使得图结构在tensorboard中更加清晰和组织化
            #定义了三个占位符，在TensorFlow计算图中预留位置以便运行时输入数据
            #接收观测值、动作编号和动作价值。[None,]表示接受任意长度一维数组
            #这第一个维度是batch维度，也就是batch size，也就是灵活的批尺寸
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")    
        #fc1 'batch normalization if any'
        layer = tf.layers.dense(
            inputs=self.tf_obs,
            units=20,
            activation=tf.nn.tanh,  # tanh activation
            name='fc1'
        )
        #fc2
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            activation=None,
            name='fc2'
        )
        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability

        with tf.name_scope('loss'):
            #负对数概率
            neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            #负对数概率乘奖励值取均值
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)#用Adam算法优化器最小化损失

    def pro_choose_action(self, observation): #select action w.r.t the actions prob
        #feed_dict是用于想计算图的占位符提供数据的机制，observation是观测值传给tf_obs占位符
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation})
        #print('pro_choose_action', prob_weights.ravel())
        #ravel()是把概率数组扁平化为一维数组。
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action
    
    def fix_choose_action(self, observation): #choose an action w.r.t max probability
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation})
        #print('fix_choose_action', prob_weights.ravel())
        action = np.argmax(prob_weights.ravel())
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_rewards()
        #print(discounted_ep_rs_norm, self.ep_rs, self.ep_obs)
        # train on episode
        self.sess.run(self.train_op, feed_dict={
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: np.array(discounted_ep_rs_norm) # shape=[None, ]
        })
    
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        
        return discounted_ep_rs_norm

    def _discount_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add
        
        #减去基线
        discounted_ep_rs -= self.reward_baseline
        self.reward_baseline = (1-self.alpha)*self.reward_baseline + self.alpha*np.mean(discounted_ep_rs)
        
        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs
    
    def save(self, checkpoint):
        saver = tf.train.Saver()
        saver.save(self.sess, checkpoint)
    
    def load(self, checkpoint):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint)
        if ckpt and ckpt.model_checkpoint_path:
            print('training from last checkpoint', checkpoint)
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        reader = pywrap_tensorflow.NewCheckpointReader(checkpoint+'trained_model.ckpt')
        #Print tensor name and values
#        var_to_shape_map = reader.get_variable_to_shape_map()
#        for key in var_to_shape_map:
#            print("tensor_name: ", key)
#            print(reader.get_tensor(key))
        self.bias_1 = reader.get_tensor('fc1/bias')
        self.kernel_1 = reader.get_tensor('fc1/kernel')
        self.bias_2 = reader.get_tensor('fc2/bias')
        self.kernel_2 = reader.get_tensor('fc2/kernel')

    def softmax(self, x):
        x = x - np.max(x)
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x)
        return softmax_x
    
    def tanh(self, x):
        exp_x = np.exp(x)
        exp_nx = np.exp(-x)
        return (exp_x - exp_nx) / (exp_x + exp_nx)
    
    def quick_time_action(self, observation): # matrix implementation for fast efficiency when the model is ready
        # l1 = self.tanh(np.dot(observation, self.kernel_1) + self.bias_1)
        l1 = np.tanh(np.dot(observation, self.kernel_1) + self.bias_1)#自己写的tanh不会处理溢出情况，numpy里的数值稳定性更高
        pro = self.softmax(np.dot(l1, self.kernel_2) + self.bias_2)
        # print('quick_time_action', pro)
        #action = np.argmax(pro)  # select action w.r.t the actions prob
        action = np.random.choice(range(self.n_actions), p=pro[0])
        return action

        
