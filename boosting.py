from getdata import getdata
from tester import tester
from xgboost import XGBClassifier
from save import save_predictions

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
    
    model = XGBClassifier(n_estimators=100,
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






if __name__ == '__main__':
    x_train, x_test, y_train, y_test, x_val = getdata()
    acc = []
    for r in range(100):

        model = boosting_trainer(x_train,y_train)
        acc.append(tester(model,x_test,y_test)[1])
    print(sum(acc)/100)
    
    #pred = model.predict(x_val) 
    #save_predictions(pred)
    
    
    
    
   
    