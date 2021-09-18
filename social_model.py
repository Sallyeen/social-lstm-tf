import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn_cell
from grid import getSequenceGridMask
import ipdb

class SocialModel():

    def __init__(self, args, infer=False):
        if infer:
            args.batch_size = 1
            args.seq_length = 1

        self.args = args
        self.infer = infer
        self.rnn_size = args.rnn_size
        self.grid_size = args.grid_size
        self.maxNumPeds = args.maxNumPeds

        with tf.name_scope("LSTM_cell"):
            cell = rnn_cell.BasicLSTMCell(args.rnn_size, state_is_tuple=False)
        
        # For each ped we have their (pedID, x, y) positions as input
        self.input_data = tf.placeholder(tf.float32, [args.seq_length, args.maxNumPeds, 3], name="input_data")
        self.target_data = tf.placeholder(tf.float32, [args.seq_length, args.maxNumPeds, 3], name="target_data")
        self.grid_data = tf.placeholder(tf.float32, [args.seq_length, args.maxNumPeds, args.maxNumPeds, args.grid_size*args.grid_size], name="grid_data")
        self.lr = tf.Variable(args.learning_rate, trainable=True, name="learning_rate")
        self.output_size = 5
        self.define_embedding_and_output_layers(args)
        # 为每位行人定义LSTM h_0\c_0
        with tf.variable_scope("LSTM_states"):
            self.initial_states = tf.split(tf.zeros([args.maxNumPeds, cell.state_size]), args.maxNumPeds, 0)

        # 为每个行人定义隐藏输出状态
        with tf.variable_scope("Hidden_states"):
            self.output_states = tf.split(tf.zeros([args.maxNumPeds, cell.output_size]), args.maxNumPeds, 0)
        
        with tf.name_scope("frame_input_data_tensors"):
            # 对于特定input，tf.split分成若干个list但不改变维度，tf.squeeze删除第零轴[seq_length]
            frame_input_data = [tf.squeeze(input_, [0]) for input_ in tf.split(self.input_data, args.seq_length, 0)]
        with tf.name_scope("frame_target_data_tensors"):
            # frame_target_data = tf.split(0, args.seq_length, self.target_data, name="frame_target_data")
            #对于特定target，删除第零轴
            frame_target_data = [tf.squeeze(target_, [0]) for target_ in tf.split(self.target_data, args.seq_length, 0)]
        with tf.name_scope("frame_grid_data_tensors"):
            # This would contain a list of tensors each of shape MNP x MNP x (GS**2) encoding the mask
            frame_grid_data = [tf.squeeze(grid_, [0]) for grid_ in tf.split(self.grid_data, args.seq_length, 0)]

        # Cost
        with tf.name_scope("Cost_related_stuff"):
            self.cost = tf.constant(0.0, name="cost")
            self.counter = tf.constant(0.0, name="counter")
            self.increment = tf.constant(1.0, name="increment")

        # 存储输出分布 参数的容器
        with tf.name_scope("Distribution_parameters_stuff"):
            self.initial_output = tf.split(tf.zeros([args.maxNumPeds, self.output_size]), args.maxNumPeds, 0)
        # 不存在行人的张量
        with tf.name_scope("Non_existent_ped_stuff"):
            nonexistent_ped = tf.constant(0.0, name="zero_ped")

        # 以序列的每个帧循环，共五次
        for seq, frame in enumerate(frame_input_data):
            print ("Frame number", seq)
            current_frame_input_data = frame  # MNP x 3 tensor 
            current_frame_grid_data = frame_grid_data[seq]  # MNP x MNP x (GS**2) tensor
            social_tensor = self.getSocialTensor(current_frame_grid_data, self.output_states)  # MNP x (GS**2 * RNN_size)         
            #以一个帧中的每个行人循环，共mnp次
            for ped in range(args.maxNumPeds):
                print ("Pedestrian Number", ped)
                pedID = current_frame_input_data[ped, 0]
                # 提取当前行人xy坐标、当前行人社会张量
                with tf.name_scope("extract_input_ped"):
                    self.spatial_input = tf.slice(current_frame_input_data, [ped, 1], [1, 2]) 
                    self.tensor_input = tf.slice(social_tensor, [ped, 0], [1, args.grid_size*args.grid_size*args.rnn_size])  # Tensor of shape (1, g*g*r)
                # 嵌入操作统一维数，input=relu(x*w+b)空间坐标1×embedding_size;社会张量1*embedding_size
                with tf.name_scope("embeddings_operations"):
                    embedded_spatial_input = tf.nn.relu(tf.nn.xw_plus_b(self.spatial_input, self.embedding_w, self.embedding_b))
                    embedded_tensor_input = tf.nn.relu(tf.nn.xw_plus_b(self.tensor_input, self.embedding_t_w, self.embedding_t_b))
                # 连接嵌入，在第一轴连接1*128
                with tf.name_scope("concatenate_embeddings"):
                   complete_input = tf.concat([embedded_spatial_input, embedded_tensor_input], 1)
                # One step of LSTM
                with tf.variable_scope("LSTM") as scope:
                    if seq > 0 or ped > 0:# 训练过一个序列或一个人之后
                        scope.reuse_variables()#验证集共享训练好的参数
                    self.output_states[ped], self.initial_states[ped] = cell(complete_input, self.initial_states[ped])
                # 应用输出线性层. 输出是一个1 x output_size的张量
                with tf.name_scope("output_linear_layer"):
                    self.initial_output[ped] = tf.nn.xw_plus_b(self.output_states[ped], self.output_w, self.output_b)
                # 提取目标数据的xy坐标。1*1
                with tf.name_scope("extract_target_ped"):
                    [x_data, y_data] = tf.split(tf.slice(frame_target_data[seq], [ped, 1], [1, 2]), 2, 1)
                # 从线性输出层的输出中提取参数
                with tf.name_scope("get_coef"):
                    [o_mux, o_muy, o_sx, o_sy, o_corr] = self.get_coef(self.initial_output[ped])
                # 计算当前行人loss
                with tf.name_scope("calculate_loss"):
                    lossfunc = self.get_lossfunc(o_mux, o_muy, o_sx, o_sy, o_corr, x_data, y_data)
                # 如果是一个不存在的ped，则不应增加cost【id为0，cost不变，counter不变；否则，cost增加，counter加一】
                with tf.name_scope("increment_cost"):
                    self.cost = tf.where(tf.equal(pedID, nonexistent_ped), self.cost, tf.add(self.cost, lossfunc))
                    self.counter = tf.where(tf.equal(pedID, nonexistent_ped), self.counter, tf.add(self.counter, self.increment))
        # Mean of the cost    
        with tf.name_scope("mean_cost"):
            self.cost = tf.div(self.cost, self.counter)
         # 获取所有训练变量
        tvars = tf.trainable_variables()
         # 用initial_states获取最终LSTM状态,mnp*state_size
        self.final_states = tf.concat(self.initial_states, 0)
        # 用initial_output获取最终分布参数
        self.final_output = self.initial_output
        # 用cost、tvars计算梯度
        self.gradients = tf.gradients(self.cost, tvars)
        # 用gradients、grad_clip裁剪梯度
        grads, _ = tf.clip_by_global_norm(self.gradients, args.grad_clip)
        # 用lr定义优化器
        optimizer = tf.train.RMSPropOptimizer(self.lr)
        # 训练操作
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        #Merge all summaries
        #merged_summary_op = tf.summary.merge_all()

    #定义变量【坐标、社会张量、输出线性层wb】
    def define_embedding_and_output_layers(self, args):
        # 为空间坐标的嵌入层定义变量
        with tf.variable_scope("coordinate_embedding"):
            self.embedding_w = tf.get_variable("embedding_w", [2, args.embedding_size], initializer=tf.truncated_normal_initializer(stddev=0.01))
            self.embedding_b = tf.get_variable("embedding_b", [args.embedding_size], initializer=tf.constant_initializer(0.01))

        # 为社会张量的嵌入层定义变量
        with tf.variable_scope("tensor_embedding"):
            self.embedding_t_w = tf.get_variable("embedding_t_w", [args.grid_size*args.grid_size*args.rnn_size, args.embedding_size], initializer=tf.truncated_normal_initializer(stddev=0.01))
            self.embedding_t_b = tf.get_variable("embedding_t_b", [args.embedding_size], initializer=tf.constant_initializer(0.01))

        # 为输出线性层定义变量
        with tf.variable_scope("output_layer"):
            self.output_w = tf.get_variable("output_w", [args.rnn_size, self.output_size], initializer=tf.truncated_normal_initializer(stddev=0.01))
            self.output_b = tf.get_variable("output_b", [self.output_size], initializer=tf.constant_initializer(0.01))

    #给定输入xy与五个参数，返回二元正态分布概率密度函数
    def tf_2d_normal(self, x, y, mux, muy, sx, sy, rho):
        # eq 3 in the paper

        normx = tf.subtract(x, mux)
        normy = tf.subtract(y, muy)
        sxsy = tf.multiply(sx, sy)
        # Calculate the exponential factor
        z = tf.square(tf.div(normx, sx)) + tf.square(tf.div(normy, sy)) - 2*tf.div(tf.multiply(rho, tf.multiply(normx, normy)), sxsy)
        negRho = 1 - tf.square(rho)
        # 分子
        result = tf.exp(tf.div(-z, 2*negRho))
        # 归一化常数
        denom = 2 * np.pi * tf.multiply(sxsy, tf.sqrt(negRho))
        # Final PDF calculation
        result = tf.div(result, denom)
        return result

    #给定五个参数，和x_data, y_data，最终返回数值稳定后的、概率密度函数的负对数似然作为  损失函数
    def get_lossfunc(self, z_mux, z_muy, z_sx, z_sy, z_corr, x_data, y_data):
        step = tf.constant(1e-3, dtype=tf.float32, shape=(1, 1))
        # 给定四个输入xy，计算四个概率密度函数
        result0_1 = self.tf_2d_normal(x_data, y_data, z_mux, z_muy, z_sx, z_sy, z_corr)
        result0_2 = self.tf_2d_normal(tf.add(x_data, step), y_data, z_mux, z_muy, z_sx, z_sy, z_corr)
        result0_3 = self.tf_2d_normal(x_data, tf.add(y_data, step), z_mux, z_muy, z_sx, z_sy, z_corr)
        result0_4 = self.tf_2d_normal(tf.add(x_data, step), tf.add(y_data, step), z_mux, z_muy, z_sx, z_sy, z_corr)

        result0 = tf.div(tf.add(tf.add(tf.add(result0_1, result0_2), result0_3), result0_4), tf.constant(4.0, dtype=tf.float32, shape=(1, 1)))
        result0 = tf.multiply(tf.multiply(result0, step), step)
        # 为了数值稳定
        epsilon = 1e-20
        result1 = -tf.log(tf.maximum(result0, epsilon))  # Numerical stability
         # Sum up all log probabilities for each data point
        return tf.reduce_sum(result1)

    #给定输出序列output，返回高斯分布五个参数
    def get_coef(self, output):
        # eq 20 -> 22 of Graves (2013)
        z = output
        # 将输出分成5个部分【第一轴分成五份】，分别对应于均值、标准差和相关系数；标准差进行幂计算，相关系数tanh
        z_mux, z_muy, z_sx, z_sy, z_corr = tf.split(z, 5, 1)
        z_sx = tf.exp(z_sx)
        z_sy = tf.exp(z_sy)
        z_corr = tf.tanh(z_corr)
        return [z_mux, z_muy, z_sx, z_sy, z_corr]

    #给定五个参数，返回满足该正态分布的一个点坐标
    def sample_gaussian_2d(self, mux, muy, sx, sy, rho):
        #从给定的二维正态分布中采样点
        mean = [mux, muy]
        cov = [[sx*sx, rho*sx*sy], [rho*sx*sy, sy*sy]]
        x = np.random.multivariate_normal(mean, cov, 1)
        return x[0][0], x[0][1]

    # 给定frame_grid_data, output_states, 返回帧中所有行人社会张量
    def getSocialTensor(self, frame_grid_data, output_states):
        #frame_grid_data : A tensor of shape MNP x MNP x (GS**2)
        #output_states : A list of tensors each of shape 1 x RNN_size of length MNP
        
        # 初始化：MNP x (GS**2) x RNN_size变成1 x (GS**2) x RNN_size of length MNP
        social_tensor = tf.zeros([self.args.maxNumPeds, self.grid_size*self.grid_size, self.rnn_size], name="social_tensor")
        social_tensor = tf.split(social_tensor, self.args.maxNumPeds, 0)
        # 把output_states在第零轴拼接，shape MNP x RNN_size
        hidden_states = tf.concat(output_states, 0)
        # 每个行人的grid_data都为MNP x (GS**2)，共MNP个
        frame_grid_ped_data = [tf.squeeze(input_, [0]) for input_ in tf.split(frame_grid_data, self.args.maxNumPeds, 0)]

        # For each pedestrian, Compute social tensor for the current pedestrian
        for ped in range(self.args.maxNumPeds):
            with tf.name_scope("tensor_calculation"):
                # (GS**2) x MNP和MNP x RNN_size=(GS**2) x RNN_size
                print (tf.transpose(frame_grid_ped_data[ped]))
                social_tensor_ped = tf.matmul(tf.transpose(frame_grid_ped_data[ped]), hidden_states)
                social_tensor[ped] = tf.reshape(social_tensor_ped, [1, self.grid_size*self.grid_size, self.rnn_size])

        # MNP x (GS**2) x RNN_size
        social_tensor = tf.concat(social_tensor, 0)
        # MNP x (GS**2 * RNN_size)
        social_tensor = tf.reshape(social_tensor, [self.args.maxNumPeds, self.grid_size*self.grid_size*self.rnn_size])
        return social_tensor

    '''#
    def sample(self, sess, traj, grid, dimensions, true_traj, num=10):
        # traj is a sequence of frames (of length obs_length)
        # so traj shape is (obs_length x maxNumPeds x 3)
        # grid is a tensor of shape obs_length x maxNumPeds x maxNumPeds x (gs**2)
        states = sess.run(self.LSTM_states)
        # For each frame in the sequence
        for index, frame in enumerate(traj[:-1]):
            data = np.reshape(frame, (1, self.maxNumPeds, 3))
            target_data = np.reshape(traj[index+1], (1, self.maxNumPeds, 3))
            grid_data = np.reshape(grid[index, :], (1, self.maxNumPeds, self.maxNumPeds, self.grid_size*self.grid_size))

            feed = {self.input_data: data, self.LSTM_states: states, self.grid_data: grid_data, self.target_data: target_data}

            [states, cost] = sess.run([self.final_states, self.cost], feed)
            # print cost

        ret = traj

        last_frame = traj[-1]

        prev_data = np.reshape(last_frame, (1, self.maxNumPeds, 3))
        prev_grid_data = np.reshape(grid[-1], (1, self.maxNumPeds, self.maxNumPeds, self.grid_size*self.grid_size))

        prev_target_data = np.reshape(true_traj[traj.shape[0]], (1, self.maxNumPeds, 3))
        # print "Prediction"
        # Prediction
        for t in range(num):
            feed = {self.input_data: prev_data, self.LSTM_states: states, self.grid_data: prev_grid_data, self.target_data: prev_target_data}
            [output, states, cost] = sess.run([self.final_output, self.final_states, self.cost], feed)
            # print cost
            # Output is a list of lists where the inner lists contain matrices of shape 1x5. The outer list contains only one element (since seq_length=1) and the inner list contains maxNumPeds elements
            # output = output[0]
            newpos = np.zeros((1, self.maxNumPeds, 3))
            for pedindex, pedoutput in enumerate(output):
                [o_mux, o_muy, o_sx, o_sy, o_corr] = np.split(pedoutput[0], 5, 0)
                mux, muy, sx, sy, corr = o_mux[0], o_muy[0], np.exp(o_sx[0]), np.exp(o_sy[0]), np.tanh(o_corr[0])
                next_x, next_y = self.sample_gaussian_2d(mux, muy, sx, sy, corr)

                newpos[0, pedindex, :] = [prev_data[0, pedindex, 0], next_x, next_y]
            ret = np.vstack((ret, newpos))
            prev_data = newpos
            prev_grid_data = getSequenceGridMask(prev_data, dimensions, self.args.neighborhood_size, self.grid_size)
            if t != num - 1:
                prev_target_data = np.reshape(true_traj[traj.shape[0] + t + 1], (1, self.maxNumPeds, 3))

        # The returned ret is of shape (obs_length+pred_length) x maxNumPeds x 3
        return ret
    '''