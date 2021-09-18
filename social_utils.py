import os
import pickle
import numpy as np
import ipdb

# data loader类：从datasets加载数据；将每个帧视为数据点，将连续帧序列视为序列。
class SocialDataLoader():

    def __init__(self, batch_size=50, seq_length=5, maxNumPeds=40, datasets=[0, 1, 2, 3, 4], forcePreProcess=False):
        '''
        Initialiser function for the SocialDataLoader class
        params:
        batch_size : Size of the mini-batch
        grid_size : Size of the social grid constructed
        forcePreProcess : Flag to forcefully preprocess the data again from csv files
        '''
        # List of data directories where raw data resides
        # self.data_dirs = ['./data/eth/univ', './data/eth/hotel',
        #                  './data/ucy/zara/zara01', './data/ucy/zara/zara02',
        #                  './data/ucy/univ']
        self.data_dirs = ['./data/eth/univ', './data/eth/hotel']

        self.used_data_dirs = [self.data_dirs[x] for x in datasets]

        # Number of datasets
        self.numDatasets = len(self.data_dirs)

        # Data directory where the pre-processed pickle file resides
        self.data_dir = './data'

        # Maximum number of peds in a single frame (Number obtained by checking the datasets)
        self.maxNumPeds = maxNumPeds

        # Store the arguments
        self.batch_size = batch_size
        self.seq_length = seq_length

        # Define the path in which the process data would be stored
        data_file = os.path.join(self.data_dir, "social-trajectories.cpkl")

        # If the file doesn't exist or forcePreProcess is true
        if not(os.path.exists(data_file)) or forcePreProcess:
            print("Creating pre-processed data from raw data")
            # Preprocess the data from the csv files of the datasets
            # Note that this data is processed in frames
            self.frame_preprocess(self.used_data_dirs, data_file)

        # Load the processed data from the pickle file
        self.load_preprocessed(data_file)
        # Reset all the data pointers of the dataloader object
        self.reset_batch_pointer()

    #给定元数据文件夹、目标搜数据文件夹，将all_frame_data, frameList_data, numPeds_data存入，需要时才会执行帧预处理
    def frame_preprocess(self, data_dirs, data_file):
        '''
        Function that will pre-process the pixel_pos.csv files of each dataset
        into data with occupancy grid that can be used
        params:
        data_dirs : List of directories where raw data resides
        data_file : The file into which all the pre-processed data needs to be stored
        '''

        # all_frame_data would be a list of numpy arrays corresponding to each dataset
        # Each numpy array would be of size (numFrames, maxNumPeds, 3) where each pedestrian's
        # pedId, x, y , in each frame is stored
        all_frame_data = []
        # frameList_data would be a list of lists corresponding to each dataset
        # Each list would contain the frameIds of all the frames in the dataset
        frameList_data = []
        # numPeds_data would be a list of lists corresponding to each dataset
        # Each list would contain the number of pedestrians in each frame in the dataset
        numPeds_data = []
        # Index of the current dataset
        dataset_index = 0

        # For each dataset
        for directory in data_dirs:

            # Define path of the csv file of the current dataset
            file_path = os.path.join(directory, 'pixel_pos.csv')

            # 从the csv file加载数据
            data = np.genfromtxt(file_path, delimiter=',')

            # 当前数据集的Frame IDs
            frameList = np.unique(data[0, :]).tolist()
            # 当前数据集中的帧总量
            numFrames = len(frameList)

            # Add the list of frameIDs to the frameList_data,提前预防处理多数据集情况
            frameList_data.append(frameList)
            # Initialize the list of numPeds for the current dataset
            numPeds_data.append([])
            # Initialize the numpy array for the current dataset，初设为全零张量
            all_frame_data.append(np.zeros((numFrames, self.maxNumPeds, 3)))

            # index to maintain the current frame
            curr_frame = 0
            for frame in frameList:
                # 提取当前帧id中的所有信息
                pedsInFrame = data[:, data[0, :] == frame]

                # Extract pedsID list,该帧行人id集合，第一个轴的第一个位置【第二行】
                pedsList = pedsInFrame[1, :].tolist()

                # Helper print statement to figure out the maximum number of peds in any frame in any dataset
                # if len(pedsList) > 1:
                # print len(pedsList)
                # DEBUG
                #    continue

                # 行人总量 in the current frame to the stored data
                numPeds_data[dataset_index].append(len(pedsList))

                # Initialize the row of the numpy array
                pedsWithPos = []

                # For each ped in the current frame可得当前帧每个人的信息
                for ped in pedsList:
                    # [0]的作用：把xy坐标变成标量
                    current_x = pedsInFrame[3, pedsInFrame[1, :] == ped][0]
                    current_y = pedsInFrame[2, pedsInFrame[1, :] == ped][0]

                    # 循环结束，表示该帧所有行人的信息 【pedID, x, y】 
                    pedsWithPos.append([ped, current_x, current_y])

                # Add the details of all the peds in the current frame to all_frame_data
                all_frame_data[dataset_index][curr_frame, 0:len(pedsList), :] = np.array(pedsWithPos)
                #print (pedsWithPos)
                #print (all_frame_data)
                # Increment the frame index，记数
                curr_frame += 1
            # Increment the dataset index
            dataset_index += 1
    
        #print ('all_frame_data:', all_frame_data)
        #print ('length of all_frame_data:', np.array(all_frame_data).shape)
        f = open(data_file, "wb")
        pickle.dump((all_frame_data, frameList_data, numPeds_data), f, protocol=2)
        f.close()

    #给定存储数据文件夹，提取data、frameList、numPedsList到 DataLoader object，并计算num_batches
    def load_preprocessed(self, data_file):
        # Load data from the pickled file
        f = open(data_file, 'rb')
        self.raw_data = pickle.load(f)
        f.close()

        # Get all the data from the pickle file
        self.data = self.raw_data[0]
        self.frameList = self.raw_data[1]
        self.numPedsList = self.raw_data[2]
        counter = 0

        # For each dataset
        for dataset in range(len(self.data)):
            # 当前数据集的帧数据：帧数据×mnp×3
            all_frame_data = self.data[dataset]
            print (len(all_frame_data))
            # Increment the counter with the number of sequences in the current dataset
            counter += int(len(all_frame_data) / (self.seq_length+2))    

        # Calculate the number of batches
        self.num_batches = int(counter/self.batch_size)

    def next_batch(self):
        # Source data
        x_batch = []
        # Target data
        y_batch = []
        # Dataset data
        d = []
        # Iteration index
        i = 0
        while i < self.batch_size:
            frame_data = self.data[self.dataset_pointer]
            idx = self.frame_pointer
            # While there is still seq_length number of frames left in the current dataset
            if idx + self.seq_length < frame_data.shape[0]:
                # All the data in this sequence
                seq_frame_data = frame_data[idx:idx+self.seq_length+1, :]
                seq_source_frame_data = frame_data[idx:idx+self.seq_length, :]
                seq_target_frame_data = frame_data[idx+1:idx+self.seq_length+1, :]
                # Number of unique peds in this sequence of frames
                pedID_list = np.unique(seq_frame_data[:, :, 0])
                numUniquePeds = pedID_list.shape[0]
                # seq_data即可得到一个序列五帧数据，以下是为了多处理一步：行人ID有序，第一行空着
                sourceData = np.zeros((self.seq_length, self.maxNumPeds, 3))
                targetData = np.zeros((self.seq_length, self.maxNumPeds, 3))
                for seq in range(self.seq_length):
                    sseq_frame_data = seq_source_frame_data[seq, :]
                    tseq_frame_data = seq_target_frame_data[seq, :]
                    for ped in range(numUniquePeds):
                        pedID = pedID_list[ped]
                        if pedID == 0:
                            continue
                        else:
                            sped = sseq_frame_data[sseq_frame_data[:, 0] == pedID, :]
                            tped = np.squeeze(tseq_frame_data[tseq_frame_data[:, 0] == pedID, :])
                            if sped.size != 0:
                                sourceData[seq, ped, :] = sped
                            if tped.size != 0:
                                targetData[seq, ped, :] = tped
                print ('sourceData:', sourceData)
                print ('targetData:', targetData)
                x_batch.append(sourceData)
                y_batch.append(targetData)
                self.frame_pointer += self.seq_length
                d.append(self.dataset_pointer)
                i += 1
            else:
                # Not enough frames left
                #表示前一数据集不够取出6个帧了，更换数据集，递增数据集指针并将帧指针设置为零
                self.tick_batch_pointer()
        return x_batch, y_batch, d

    #Advance the dataset pointer
    def tick_batch_pointer(self):
        # Go to the next dataset
        self.dataset_pointer += 1
        # Set the frame pointer to zero for the current dataset
        self.frame_pointer = 0
        # If all datasets are done, then go to the first one again
        if self.dataset_pointer >= len(self.data):
            self.dataset_pointer = 0

    #Reset all pointers
    def reset_batch_pointer(self):
        # Go to the first frame of the first dataset
        self.dataset_pointer = 0
        self.frame_pointer = 0
