from sklearn.dummy import DummyClassifier
from tester import tester
from getdata import getdata
from save import save_predictions
def naive_trainer(features,label):
    model = DummyClassifier(strategy='most_frequent')
    model.fit(features, label)
    return model




if __name__ == '__main__':
    x_train, x_test, y_train, y_test,x_val = getdata()
    model = naive_trainer(x_train, y_train)
    print(tester(model,x_test,y_test))
    save_predictions(model.predict(x_val))