from getdata import getdata
from tester import tester
from sklearn.tree import ExtraTreeClassifier
from save import save_predictions
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

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
    
    model =  AdaBoostClassifier(n_estimators=2000, random_state=0,learning_rate=0.1)
    print(model.get_params())
    model.fit(features,label)
    
    return model









def boosting_tuner(features,label):
    params = {
        
        
        'learning_rate': [0.01,0.1,0.25,0.5,0.75,1],
         'n_estimators': [10,100,500,1000,2000,5000]
        
        }
    
    model = AdaBoostClassifier(random_state=0)
    grid = GridSearchCV(model,param_grid=params,verbose=3,n_jobs=-1,scoring='balanced_accuracy',cv=5)
    grid.fit(features,label)
    
    return grid
    
    
    
    
   
    

if __name__ == '__main__':
    tune = False

    x_train, x_test, y_train, y_test, x_val = getdata()

    if tune:
        grid = boosting_tuner(x_train,y_train)
    
        print('\n Best estimator:')
        print(grid.best_estimator_)
        print('\n Best hyperparameters:')
        print(grid.best_params_)
        model = grid.best_estimator_
        print(tester(model,x_test,y_test))
        #pred = model.predict(x_val) 
        #save_predictions(pred)
    
    else:        
        model = boosting_trainer(x_train,y_train)
        print(tester(model,x_test,y_test))
        
        #pred = model.predict(x_val) 
        #save_predictions(pred)
        
    
    
    
   
    