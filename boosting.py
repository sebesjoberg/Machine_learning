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


def boosting_trainer(features,label,x_test,y_test):
    
    model = XGBClassifier(n_estimators=10000,
                          objective="binary:logistic",random_state=42,early_stopping_rounds=10, scale_pos_weight=get_weights(label),
                          subsample=0.5, eval_metric="auc",
                          learning_rate=0.5)
    
    model.fit(features,label, eval_set=[(x_test,y_test)])
    
    return model






if __name__ == '__main__':
    x_train, x_test, y_train, y_test, x_val = getdata()
   
    model = boosting_trainer(x_train,y_train,x_test,y_test)
    print(tester(model,x_test,y_test))
    
    pred = model.predict(x_val) 
    save_predictions(pred)
    
    
    
    
   
    