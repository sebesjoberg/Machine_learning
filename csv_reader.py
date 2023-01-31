import pandas as pd
def csv_reader(filename):
    data = pd.read_csv(filename)
    return data


if __name__ == '__main__':
    print(csv_reader('data/train.csv'))