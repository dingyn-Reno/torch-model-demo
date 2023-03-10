import os
import pandas as pd
import numpy as np
import json
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision
from tqdm import tqdm

class CFG:
    data_path = '/share/kaggle/asl-signs/'
    quick_experiment = False
    is_training = True
    use_aggregation_dataset = True
    num_classes = 250
    rows_per_frame = 543
    batch_size=128
    epochs = 150

def load_relevant_data_subset_with_imputation(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    data.replace(np.nan, 0, inplace=True)
    n_frames = int(len(data) / CFG.rows_per_frame)
    data = data.values.reshape(n_frames, CFG.rows_per_frame, len(data_columns))
    return data.astype(np.float32)

def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / CFG.rows_per_frame)
    data = data.values.reshape(n_frames, CFG.rows_per_frame, len(data_columns))
    return data.astype(np.float32)

def read_dict(file_path):
    path = os.path.expanduser(file_path)
    with open(path, "r") as f:
        dic = json.load(f)
    return dic

class Feeder(Dataset):
    def __init__(self,data_path='/share/kaggle/asl-signs/train.csv',label_path='/share/kaggle/asl-signs/sign_to_prediction_index_map.json',debug=False, mode='train',fold=0):
        self.debug=debug
        self.data_path=data_path
        self.label_path=label_path
        self.mode=mode
        self.fold=fold
        self.load(fold,mode)
    def __len__(self):
        return len(self.y)

    def load(self,fold,mode):
        train=pd.read_csv(self.data_path)
        label_index = read_dict(self.label_path)
        index_label = dict([(label_index[key], key) for key in label_index])
        train["label"] = train["sign"].map(lambda sign: label_index[sign])
        if CFG.use_aggregation_dataset == False:
            xs = []
            ys = []
            num_frames = np.zeros(len(train))
            for i in tqdm(range(len(train))):
                path = f"{CFG.data_path}{train.iloc[i].path}"
                data = load_relevant_data_subset_with_imputation(path)
                ## Mean Aggregation
                xs.append(np.mean(data, axis=0))
                ys.append(train.iloc[i].label)
                num_frames[i] = data.shape[0]
                if CFG.quick_experiment and i == 4999:
                    break
            ## Save number of frames of each training sample for data analysis
            train["num_frames"] = num_frames
            X = np.array(xs)
            y = np.array(ys)
            # print(train["num_frames"].describe())
            train.to_csv("train.csv", index=False)
            np.save("X.npy",X)
            np.save("y.npy",y)
        else:
            X = np.load("/home/dingyuning/GISLR/feeder/X.npy")
            y = np.load("/home/dingyuning/GISLR/feeder/y.npy")

        if self.fold==-1:
            self.X=X
            self.y=y
        else:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            if mode=='train':
                self.X=X_train
                self.y=y_train
            else:
                self.X=X_test
                self.y=y_test




    def __getitem__(self, idx):
        data=self.X[idx]
        label=self.y[idx]
        return data,label,idx

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

if __name__=='__main__':
    feeder=Feeder()
    print(len(feeder))
    print(feeder[0][0].shape)
