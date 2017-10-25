import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.optim import Adam
import torch.nn.functional as F

def min_max_datedata(date, reverse=False):
    if not reverse:
        y = (date[0] - 2011)/(2018-2011)
        m = (date[1] - 8)/(12-8)
        d = (date[2] - 1)/(31-1)
    else:
        y = date[0]*(2018-2011) + 2011
        m = date[1]*(12 - 8) + 8
        d = date[2]*(31 - 1) + 1
    return np.array([y,m,d]).astype('float')

def min_max_tempdata(temps, reverse=False):
    if not reverse:
        max = (temps[0] + 20)/(50 + 20)
        min = (temps[1] + 20)/(50 + 20)
    else:
        max = temps[0]*(50+20) - 20
        min = temps[1]*(50 + 20) - 20
    return np.array([max, min]).astype('float')


class WeatherDataset(Dataset):

    def __init__(self, csv_file):
        data_frame = pd.read_csv(csv_file)

        self.data_tensor = data_frame[['year','month','day']].as_matrix()
        self.target_tensor = data_frame[['max','min']].as_matrix()



    def __len__(self):
        return len(self.data_tensor)

    def __getitem__(self, item):
        return min_max_datedata(self.data_tensor[item]), min_max_tempdata(self.target_tensor[item])


class WeatherNet(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(WeatherNet, self).__init__()

        self.layer1 = torch.nn.Linear(D_in, 30)
        self.layer2 = torch.nn.Linear(30, 6)
        self.layer3 = torch.nn.Linear(6, D_out)

    def forward(self, X):
        y_ = F.relu(self.layer1(X))
        y_ = F.relu(self.layer2(y_))
        y_ = F.relu(self.layer3(y_))

        return y_

def train():
    D_in, D_out, batch = 3, 2, 16

    ds = WeatherDataset("weather_data.csv")
    dl = DataLoader(ds, batch_size=batch, shuffle=True, pin_memory=False, drop_last=True)

    net = WeatherNet(D_in, D_out)

    criteria = torch.nn.MSELoss()
    optimizer = Adam(net.parameters())
    print(net)
    dtype = torch.FloatTensor

    for epoch in range(10):
        running_loss = 0.0

        for i, data in enumerate(dl,0):
            X, y = data

            X, y = Variable(X.type(dtype)), Variable(y.type(dtype))

            optimizer.zero_grad()

            y_ = net(X)
            loss = criteria(y_, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            if i % 20 == 19:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

    print('Training finished')

    pred = net(Variable(torch.from_numpy(min_max_datedata([2017,10,15]))).type(dtype))
    # print(min_max_tempdata(pred, True))
    print(min_max_tempdata(pred.data.numpy(),True))

train()