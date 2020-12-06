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


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data,result, test_size=0.33, random_state=42)
import seaborn as sns
sns.countplot(result)

X_train = torch.from_numpy(X_train.to_numpy()).float()
y_train = torch.squeeze(torch.from_numpy(y_train.to_numpy()).float())
X_test = torch.from_numpy(X_test.to_numpy()).float()
y_test = torch.squeeze(torch.from_numpy(y_test.to_numpy()).float())
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

model=CreditCardNN(X_train.shape[1],10,10,1)
criterion = nn.BCELoss()
from torch import nn, optim

optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)
model = model.to(device)
criterion = criterion.to(device)


def calculate_accuracy(y_true, y_pred):
  predicted = y_pred.ge(.5).view(-1)
  return (y_true == predicted).sum().float() / len(y_true)

def round_tensor(t, decimal_places=3):
  return round(t.item(), decimal_places)

for epoch in range(1000):
    y_pred = model(X_train)
    y_pred = torch.squeeze(y_pred)
    train_loss = criterion(y_pred, y_train)
    if epoch % 100 == 0:
      train_acc = calculate_accuracy(y_train, y_pred)
      y_test_pred = model(X_test)
      y_test_pred = torch.squeeze(y_test_pred)
      test_loss = criterion(y_test_pred, y_test)
      test_acc = calculate_accuracy(y_test, y_test_pred)
      print(
f'''epoch {epoch}
Train set - loss: {round_tensor(train_loss)}, accuracy: {round_tensor(train_acc)}
Test  set - loss: {round_tensor(test_loss)}, accuracy: {round_tensor(test_acc)}
''')
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()




MODEL_PATH = '/Users/gauravtyagi/Downloads/model'
torch.save(model, MODEL_PATH)
model = torch.load(MODEL_PATH)
