# Author: Apala Thakur
# WebQA JSON file readers

import json
def read_train_val(filename):
    with open(filename,'r') as f:
        data = json.load(f)
    train = []
    val = []
    
    for k,v in data.items():
        
        if v['split'] == 'val':
            val.append(v)
            
        else:
            train.append(v)
    return train,val

def read_test(filename):
    with open(filename,'r') as f:
        data = json.load(f)
    
    test = []
    for k,v in data.items():
         test.append(v)
    return test
            