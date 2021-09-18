import numpy as np
import tensorflow as tf

import os
import pickle
import argparse
import ipdb

from social_utils import SocialDataLoader
from social_model import SocialModel
from grid import getSequenceGridMask
# from social_train import getSocialGrid, getSocialTensor

#计算预测轨迹和真实轨迹之间的平均欧氏距离误差，返回误差分量-每一预测时刻的误差
def get_mean_error(predicted_traj, true_traj, observed_length, maxNumPeds):
    
    # 存储所有误差的数据结构，长度为预测轨迹长度
    error = np.zeros(len(true_traj) - observed_length)
    # 对于轨迹预测的每个点【time step】,送入一帧
    for i in range(observed_length, len(true_traj)):
        # 预测的位置 maxNumPeds x 3 matrix
        pred_pos = predicted_traj[i, :]
        # 真实位置 maxNumPeds x 3 matrix
        true_pos = true_traj[i, :]
        timestep_error = 0
        counter = 0
        #对于预测轨迹某个点的每个人，累计得到这一帧所有人的预测误差
        for j in range(maxNumPeds):
            if true_pos[j, 0] == 0:
                # Non-existent ped
                continue
            else:
                timestep_error += np.linalg.norm(true_pos[j, [1, 2]] - pred_pos[j, [1, 2]])
                counter += 1

        error[i - observed_length] = timestep_error / counter

        # The euclidean distance is the error
        # error[i-observed_length] = np.linalg.norm(true_pos - pred_pos)

    # Return the mean error
    return np.mean(error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obs_length', type=int, default=5,
                        help='Observed length of the trajectory')
    parser.add_argument('--pred_length', type=int, default=3,
                        help='Predicted length of the trajectory')
    parser.add_argument('--test_dataset', type=int, default=0,
                        help='Dataset to be tested on')

    sample_args = parser.parse_args()

    with open(os.path.join('save', 'social_config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)

    model = SocialModel(saved_args, True)
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()

    # Get the checkpoint state for the model
    ckpt = tf.train.get_checkpoint_state('save')
    print ('loading model: ', ckpt.model_checkpoint_path)

    # Restore the model at the checkpoint
    saver.restore(sess, ckpt.model_checkpoint_path)

    # Dataset to get data from
    dataset = [sample_args.test_dataset]

    # batch_size 1 and seq_length equal to observed_length + pred_length
    data_loader = SocialDataLoader(1, sample_args.pred_length + sample_args.obs_length, saved_args.maxNumPeds, dataset, True)

    data_loader.reset_batch_pointer()

    total_error = 0
    
    # 每个batch
    for b in range(data_loader.num_batches):
        # Get the source, target and dataset data for the next batch
        x, y, d = data_loader.next_batch()

        # Batch size is 1
        x_batch, y_batch, d_batch = x[0], y[0], d[0]

        if d_batch == 0 and dataset[0] == 0:
            dimensions = [640, 480]
        else:
            dimensions = [720, 576]

        #32,2
        grid_batch = getSequenceGridMask(x_batch, dimensions, saved_args.neighborhood_size, saved_args.grid_size)

        obs_traj = x_batch[:sample_args.obs_length]
        obs_grid = grid_batch[:sample_args.obs_length]
        # obs_traj is an array of shape obs_length x maxNumPeds x 3

        complete_traj = model.sample(sess, obs_traj, obs_grid, dimensions, x_batch, sample_args.pred_length)

        # ipdb.set_trace()
        # complete_traj is an array of shape (obs_length+pred_length) x maxNumPeds x 3
        total_error += get_mean_error(complete_traj, x[0], sample_args.obs_length, saved_args.maxNumPeds)

        print ("Processed trajectory number : ", b, "out of ", data_loader.num_batches, " trajectories")

    # Print the mean error across all the batches
    print ("Total mean error of the model is ", total_error/data_loader.num_batches)

if __name__ == '__main__':
    main()
