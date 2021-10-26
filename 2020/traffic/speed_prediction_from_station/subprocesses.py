import subprocess

import threading, requests, time
 
def getHtml(url):
    resp = requests.get(url)
    time.sleep(1)
    print(url, len(resp.text), ' chars')
 


import myconfig

model_list = myconfig.models.keys()
dataset = myconfig.dataset

def testModel(i, model):
    print(' '.join(["/newmnt/miniconda3/envs/GIS/bin/python", "train_model.py", str(i), model, dataset]))
    subprocess.run(["/newmnt/miniconda3/envs/GIS/bin/python", "train_model.py", str(i), model, dataset])


import time
for i, model in enumerate(model_list):
    t1 = threading.Thread(target=testModel, args=((i+myconfig.gpu)%8,model,))
    t1.daemon = True 
    t1.start()
    time.sleep(10)
