def save_predictions(pred):
    out = ""
    for y in pred:
        
        out += str(y)+","
    out = out[:-1]
    
    text_file = open("predictions.csv", "w")
 
    text_file.write(out)
 
    text_file.close()