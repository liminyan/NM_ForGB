from sklearn.svm import SVR,LinearSVR
# import torch
# import torch.nn as nn
import time
import numpy as np
import multiprocessing
import time
import mpi4py.MPI as MPI
import DataProcess
import os
from pathlib import Path
from sklearn.externals import joblib
from sklearn import preprocessing

from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem


from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


class Model(object):
    """
    <<interface>> Model，接口类
    """

    def fit(self):
        """
        模型训练
        """
        pass

    def predict(self):
        """
        模型预测
        """
        pass

    def save(self, path):
        """
        模型保存
        """
        pass

    def load(self, path):
        """
        模型加载
        """
        pass

class SupportVectorRegression(Model):
    """
    用于集合后处理的超级集合方法，预测模型为支持向量回归
    Attributes:
        kernel: SVR 参数，可参考https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR
    """
    def __init__(self, kernel='rbf', degree= 36,gamma='auto', coef0=0.2, tol=0.001, C=20,
                 epsilon=0.01, shrinking=True, cache_size=200, verbose=False, max_iter=-1):
        """
        初始化实例属性，SVR 模型参数为默认值 
        """
        self.kernel_models = {}
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.C = C
        self.epsilon = epsilon
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.tar = ''
        self.line_num = 0

    def fit(self, TrainData,partial_fit = 'svr'):
        """
        模型训练
        :param TrainData: 数据处理器实例，主要使用其中的 x_train 和 y_train
        :return: 返回实例本身
        """

        comm = MPI.COMM_WORLD
        comm_rank = comm.Get_rank()
        comm_size = comm.Get_size()
        max_size = 180 * 360
        per_size = int(max_size/comm_size)
        begin = comm_rank*per_size
        end = (comm_rank + 1)*per_size
        s = time.time()

        if TrainData.data_label['x_train'] and TrainData.data_label['y_train']:
            start = time.time()
            # 逐格点训练模型， 保存到 kernel_models 字典
            for i in range(begin,end):
                x = int (i / 360)
                y = int (i % 360)
                key = str(x)+"_"+str(y)
           
                if not self.kernel_models.__contains__(key):
                    if partial_fit == 'line':
                        # reg = SGDRegressor(max_iter=1000, tol=1e-3)
                        reg = make_pipeline(StandardScaler(),
                        SGDRegressor(max_iter=1000, tol=1e-3))
                    elif partial_fit == 'mlp':
                        reg = make_pipeline(StandardScaler(),
                        MLPRegressor(random_state=1, max_iter=500))
                        # X, y = make_regression(n_samples=200, random_state=1)
                       
                    elif partial_fit == 'svr':
                        reg = make_pipeline(StandardScaler(),SVR(kernel=self.kernel, degree=self.degree, 
                        gamma= self.gamma,
                        coef0=self.coef0, tol=self.tol, C=5, epsilon=self.epsilon,
                        shrinking=self.shrinking, 
                        # cache_size=self.cache_size,
                        verbose=self.verbose, max_iter=self.max_iter))
                else:
                    reg = self.kernel_models[key]

                in_ = TrainData.x_train[:,:,i - begin]
                out_ = TrainData.y_train[:,0,i - begin]

# def partial_pipe_fit(pipeline_obj, df):
# X = pipeline_obj.named_steps['mapper'].fit_transform(df)
# Y = df['class']
# pipeline_obj.named_steps['clf'].partial_fit(X,Y)
# 
# {'standardscaler': StandardScaler(copy=True, with_mean=True, with_std=True), 
#  'sgdregressor': SGDRegressor(alpha=0.0001, average=False, epsilon=0.1, eta0=0.01,
#        fit_intercept=True, l1_ratio=0.15, learning_rate='invscaling',
#        loss='squared_loss', max_iter=1000, n_iter=None, penalty='l2',
#        power_t=0.25, random_state=None, shuffle=True, tol=0.001, verbose=0,
#        warm_start=False)}
                if partial_fit == 'line':
                    in_ = reg.named_steps['standardscaler'].fit_transform(in_)
                    reg.named_steps['sgdregressor'].partial_fit(in_,out_)
                if partial_fit == 'svr':
                    reg.fit(in_,out_)
                if partial_fit == 'mlp':
                    in_ = reg.named_steps['standardscaler'].fit_transform(in_)
                    reg.named_steps['mlpregressor'].partial_fit(in_,out_)
                    # reg.partial_fit(in_,out_)

                self.kernel_models[key] = reg


            elapsed = (time.time() - start)
            print("train time used:",round(elapsed,2))

            return self
        else:
            print('Training dataset is not available!')
            return self



    def predict(self, TrainData):
        """
        模型预测
        :param TrainData: 数据处理器实例，主要使用其中的 x_test
        :return: 返回预测值或实例本身
        """
        comm = MPI.COMM_WORLD
        comm_rank = comm.Get_rank()
        comm_size = comm.Get_size()
        max_size = 180 * 360
        per_size = int(max_size/comm_size)
        begin = comm_rank*per_size
        end = (comm_rank + 1)*per_size

        print(max_size,comm_size,per_size)
        print(TrainData.x_test.shape)
        print(TrainData.y_test.shape)

        if TrainData.data_label['x_test']:
            start = time.time()
            sample_num = TrainData.x_test.shape[0]
            results = np.zeros((sample_num, TrainData.lats, TrainData.lons))
            total_res = np.empty([end - begin,sample_num],dtype = np.float64)
            label_res = np.empty([end - begin,sample_num],dtype = np.float64)
            input_res = np.empty([end - begin,sample_num],dtype = np.float64)

            for i in range(begin,end):
                x = int (i / 360)
                y = int (i % 360)
                key = str(x)+"_"+str(y)
                reg = self.kernel_models[str(x)+'_'+str(y)]

                total_res[i-begin] = reg.predict(TrainData.x_test[:,:,i-begin])
                label_res[i-begin] = TrainData.y_test[:,0,i-begin]
                input_res[i-begin] = TrainData.x_test[:,self.line_num,i-begin]

                # max_res = np.max(input_res)
                # min_res = np.min(input_res)

                # max_label_res = np.max(label_res)
                # min_label_res = np.min(label_res)

                # print(max_res,min_res)
                # print(max_label_res,min_label_res)

                # total_res[i-begin] = np.where(total_res[i-begin]>max_res,max_res,total_res[i-begin])
                # total_res[i-begin] = np.where(total_res[i-begin]<min_res,min_res,total_res[i-begin])


            # print('|>x',total_res.shape)
            # print('|>y',label_res.shape)

            if comm_rank == 0:
                recv_buf = np.empty([max_size,sample_num],dtype = np.float64)
                test_buf = np.empty([max_size,sample_num],dtype = np.float64)

            else:
                recv_buf = None
                test_buf = None

            comm.Gather(total_res,recv_buf,root = 0)
            comm.Gather(label_res,test_buf,root = 0)
            elapsed = (time.time() - start)
            print("predict time used:",round(elapsed,2))

            if comm_rank == 0:

                recv_buf = (recv_buf.T).reshape(sample_num, TrainData.lats, TrainData.lons)
                TrainData.y_test = (test_buf.T).reshape(sample_num, TrainData.lats, TrainData.lons)

                return recv_buf
            else:
                return total_res
        else:
            print('Predicting dataset is not available!')
            return self

    def save(self, path):
        """
        模型保存
        :param path: 模型保存的路径，保存的文件为 .npy 格式，如 0302_PRAVG_SVR.npy
        :return: 返回实例本身
        """
        comm = MPI.COMM_WORLD
        comm_rank = comm.Get_rank()
        comm_size = comm.Get_size()
        if path[-4:] == '.npy': 
            path_dir = path[:-4]
        else:
            path_dir = path

        my_dir = Path(path_dir)
        my_file = Path(path)

        if not my_dir.exists():
            os.makedirs(path_dir)
        print(path_dir)
        print(path)

        max_size = 180 * 360
        per_size = int(max_size/comm_size)
        begin = comm_rank*per_size
        end = (comm_rank + 1)*per_size
        s = time.time()
        
        if my_dir.exists() :
            # kernel_models = np.load(path, allow_pickle=True).item()
            # self.kernel_models = {}
            for i in range(begin,end):
                x = int (i / 360)
                y = int (i % 360)
                key = str(x)+"_"+str(y)
                joblib.dump(self.kernel_models[key], path_dir+'/'+key+'.m')
                modelmath = path_dir+'/'+key+'.m'

        print(comm_rank,'model num',len(self.kernel_models.keys()))
        e = time.time()
        print(comm_rank,'saveModel time',e-s)
        return self


    def load(self, path):
        """
        模型加载
        :param path: 加载模型的路径，加载的文件为 .npy 格式，如 0302_PRAVG_SVR.npy
        :return: 返回实例本身
        """
        comm = MPI.COMM_WORLD
        comm_rank = comm.Get_rank()
        comm_size = comm.Get_size()
        if path[-4:] == '.npy': 
            path_dir = path[:-4]
        else:
            path_dir = path

        my_dir = Path(path_dir)
        my_file = Path(path)

        if not my_dir.exists():
            os.makedirs(path_dir)

        max_size = 180 * 360
        per_size = int(max_size/comm_size)
        begin = comm_rank*per_size
        end = (comm_rank + 1)*per_size
        s = time.time()
        
        if my_dir.exists() :
            # kernel_models = np.load(path, allow_pickle=True).item()
            self.kernel_models = {}
            for i in range(begin,end):
                x = int (i / 360)
                y = int (i % 360)
                key = str(x)+"_"+str(y)
                # ( path_dir+self.tar+'/'+key+'.m')
                # joblib.dump(self.kernel_models[key],path_dir+'/'+key+'.m' )
                self.kernel_models[key] = joblib.load(path_dir+'/'+key+'.m')
        print(comm_rank,'model num',len(self.kernel_models.keys()))
        e = time.time()
        print(comm_rank,'loadModel time',e-s)
        return self

