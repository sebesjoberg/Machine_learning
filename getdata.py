
from csv_reader import csv_reader
from minmax import minmax
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def getdata():
    
    df = csv_reader('data/train.csv')
    label = df.get('Lead')
    df = df.drop('Lead', axis=1)
    #df = df.apply(minmax, axis=0)
    df = pd.DataFrame(MinMaxScaler().fit_transform(df.values), columns=df.columns, index=df.index)
    
    data_classes = ["Male","Female"]
    
    x_train, x_test, y_train, y_test = train_test_split(df, label,train_size=0.7, random_state = 0)
    
    
    y_train, y_test = y_train.apply(data_classes.index), y_test.apply(data_classes.index)
    x_val = csv_reader('data/test.csv')
    x_val = pd.DataFrame(MinMaxScaler().fit_transform(x_val.values), columns=x_val.columns, index=x_val.index)
    
    
    return x_train, x_test, y_train, y_test, x_val

if __name__ == "__main__":
    print(getdata())