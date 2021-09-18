'''
Helper functions to compute the masks relevant to social grid

Author : Anirudh Vemula
Date : 29th October 2016
'''
import numpy as np

#给定frame、dimension、邻域尺寸、网格尺寸，可得三维矩阵，考虑i个人的邻居j的话，k=1
def getGridMask(frame, dimensions, neighborhood_size, grid_size):
    '''
    计算二进制掩码：每个ped在其他ped中的占据情况
    This function computes the binary mask that represents the
    occupancy of each ped in the other's grid
    params:
    frame : This will be a MNP x 3 matrix with each row being [pedID, x, y]
    dimensions : This will be a list [width, height]
    neighborhood_size : Scalar value representing the size of neighborhood considered
    grid_size : Scalar value representing the size of the grid discretization
    '''
    mnp = frame.shape[0]
    width, height = dimensions[0], dimensions[1]
    #frame_mask是一个三维张量
    frame_mask = np.zeros((mnp, mnp, grid_size**2))
    #划分小网格的宽和高
    width_bound, height_bound = neighborhood_size/(width*1.0), neighborhood_size/(height*1.0)
    # For each ped in the frame (existent and non-existent)
    for pedindex in range(mnp):
        # If pedID is zero, then non-existent ped
        if frame[pedindex, 0] == 0:
            # Binary mask should be zero for non-existent ped
            continue
        # Get x and y of the current ped
        current_x, current_y = frame[pedindex, 1], frame[pedindex, 2]
        width_low, width_high = current_x - width_bound/2, current_x + width_bound/2
        height_low, height_high = current_y - height_bound/2, current_y + height_bound/2
        # For all the other peds
        for otherpedindex in range(mnp):
            # 行人不存在
            if frame[otherpedindex, 0] == 0:
                # Binary mask should be zero
                continue
            # 行人为当前考虑行人
            if frame[otherpedindex, 0] == frame[pedindex, 0]:
                # The ped cannot be counted in his own grid
                continue
            # 符合条件行人的坐标
            other_x, other_y = frame[otherpedindex, 1], frame[otherpedindex, 2]
            # 坐标不在规定范围内
            if other_x >= width_high or other_x < width_low or other_y >= height_high or other_y < height_low:
                continue
            # 计算栅格单元，np.floor只是向下取整不改变数据类型，需要转int
            cell_x = int(np.floor(((other_x - width_low)/width_bound) * grid_size))
            cell_y = int(np.floor(((other_y - height_low)/height_bound) * grid_size))
            # Other ped is in the corresponding grid cell of current ped
            frame_mask[pedindex, otherpedindex, cell_x + cell_y*grid_size] = 1
    return frame_mask

def getSequenceGridMask(sequence, dimensions, neighborhood_size, grid_size):
    '''
    Get the grid masks for all the frames in the sequence
    params:
    sequence : A numpy matrix of shape SL x MNP x 3
    dimensions : This will be a list [width, height]
    neighborhood_size : Scalar value representing the size of neighborhood considered
    grid_size : Scalar value representing the size of the grid discretization
    '''
    sl = sequence.shape[0]
    mnp = sequence.shape[1]
    sequence_mask = np.zeros((sl, mnp, mnp, grid_size**2))
    for i in range(sl):
        sequence_mask[i, :, :, :] = getGridMask(sequence[i, :, :], dimensions, neighborhood_size, grid_size)
    return sequence_mask
