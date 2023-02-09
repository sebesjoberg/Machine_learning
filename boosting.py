from getdata import getdata
from tester import tester
from xgboost import XGBClassifier
from save import save_predictions
from sklearn.model_selection import GridSearchCV
#0=male 1=female
def get_weights(label):
    zero = 0
    one = 0
    for l in label:
        if l == 0:
            zero +=1
        else:
            one +=1
   
    return zero/one


def boosting_trainer(features,label):
    
    model = XGBClassifier(n_estimators=1000,
                          objective="binary:logistic", scale_pos_weight=get_weights(label),
                          booster = 'dart',
                          subsample=0.75, 
                          learning_rate=0.1,
                          colsample_bytree=0.75,
                          gamma=0.1,max_delta_step=1,
                          max_depth=7,
                          min_child_weight=1,
                          )
    
    model.fit(features,label)
    
    return model









def boosting_tuner(features,label):
    params = {
        'booster':['gbtree','dart'],
        'min_child_weight': [0.1, 0.5, 1.0],#0-inf
        'gamma': [0.1, 0.5, 1,],#0-inf
        'subsample': [0.5, 0.75, 1.0],#0-1
        'colsample_bytree': [0.5, 0.75, 1.0],#0-1
        'max_depth': [3,5,7],#1.inf
        'learning_rate': [0.01,0.1,1.0],#0-1
        'max_delta_step': [0, 0.5, 1]
        
        }
    
    model = XGBClassifier(n_estimators=100,
                          objective="binary:logistic",random_state=0, scale_pos_weight=get_weights(label)
                          
                          )
    grid = GridSearchCV(model,param_grid=params,verbose=3,n_jobs=-1,scoring='accuracy')
    grid.fit(features,label)
    
    return grid
    
    
    
    
   
    

if __name__ == '__main__':
    tune = False

    x_train, x_test, y_train, y_test, x_val = getdata()

    if tune:
        grid = boosting_tuner(x_train,y_train)
    
        print('\n Best estimator:')
        print(grid.best_estimator_)
    
        print(grid.best_score_ * 2 - 1)
        print('\n Best hyperparameters:')
        print(grid.best_params_)
        model = grid.best_estimator_
        print(tester(model,x_test,y_test))
        #pred = model.predict(x_val) 
        #save_predictions(pred)
    
    else:        
        model = boosting_trainer(x_train,y_train)
        print(tester(model,x_test,y_test))
        
        pred = model.predict(x_val) 
        save_predictions(pred)
        
    
    
    
   
    