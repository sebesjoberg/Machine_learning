from sklearn.metrics import accuracy_score, balanced_accuracy_score
def tester(model,features,labels):
    
    predicted = model.predict(features)

    return "accuracy", accuracy_score(labels, predicted)*100, "balanced accuracy",balanced_accuracy_score(labels, predicted)*100