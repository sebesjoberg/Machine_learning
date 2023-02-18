from sklearn.preprocessing import MinMaxScaler
def minmax(s):
    return MinMaxScaler().fit_transform(s)