# https://www.kaggle.com/sakshigoyal7/credit-card-customers

import pandas as pd
from sklearn.preprocessing import LabelEncoder

file=pd.read_csv("/Users/gauravtyagi/Downloads/data/BankChurners.csv")
print(file)
labelEncoder = LabelEncoder()
file['Attrition_Flag']=labelEncoder.fit_transform(file['Attrition_Flag'])
file['Marital_Status']=labelEncoder.fit_transform(file['Marital_Status'])
file['Gender']=labelEncoder.fit_transform(file['Gender'])
file['Education_Level']=labelEncoder.fit_transform(file['Education_Level'])
file['Income_Category']=labelEncoder.fit_transform(file['Income_Category'])
file['Card_Category']=labelEncoder.fit_transform(file['Card_Category'])


data=file
data=data.drop(data.columns[[0,1]], axis=1)
result=file['Attrition_Flag']
print(result)




import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable


class CreditCardNN(nn.Module):

    def __init__(self, input_size, hidden1_size, hidden2_size, num_classes):
        super(CreditCardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out
