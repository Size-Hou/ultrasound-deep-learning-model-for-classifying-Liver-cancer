import os
from Model.resnet18 import ResNet, train_model
from LoadData.LoadData import InputTrainImg, InputTestImg

btch = 32
epoch = 1000
width = 128
num_task = 3

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

datasets = InputTrainImg('D:/projects/Liver/data', width)
train_x, train_y, test_x, test_y, test_z= datasets.load_train_data()
# dat = InputTestImg('D:/projects/Liver/test', width)
# test_x, test_y = dat.load_test_data()
# train_model(train_x, train_y, test_x, test_y, num_task, epoch, btch, width)
