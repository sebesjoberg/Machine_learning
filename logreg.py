from getdata import getdata
import numpy as np
import sklearn.linear_model as skl_lm
from sklearn.model_selection import GridSearchCV
from tester import tester
def model_trainer(x_train, y_train):
    model = skl_lm.LogisticRegression(solver='lbfgs')
    model.fit(x_train, y_train)


    return model


def model_tuner(x_train,y_train):
    model = skl_lm.LogisticRegression(solver='saga',random_state=0,max_iter=10000)
    params = {
        #'penalty': ['l1','l2'],
        'penalty': ['elasticnet'],
        'l1_ratio': np.linspace(0,1,num=30),
        'C': np.logspace(-2, 3, 20)
        
        }
    
    
    grid = GridSearchCV(model,param_grid=params,verbose=3,n_jobs=-1,scoring='accuracy',cv=5)
    grid.fit(x_train,y_train)
    return grid
if __name__ == "__main__":
    tune = True
    x_train, x_test, y_train, y_test, x_val = getdata()
    if tune:
        grid = model_tuner(x_train, y_train)
        print('\n Best estimator:')
        print(grid.best_estimator_)
        print('\n Best hyperparameters:')
        print(grid.best_params_)
        model = grid.best_estimator_
        print(tester(model,x_test,y_test))
    else:
        
        model = model_trainer(x_train, y_train)
        print(tester(model,x_test,y_test))