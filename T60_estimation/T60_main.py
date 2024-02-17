
import torch
from torch.utils.data import DataLoader
import numpy as np
from collections import Counter
from model import Net
import re

from utils import prepare_data, extract_wav_paths, extract_p_paths
#%% device and model initialization
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# Load the trained model
model = Net()
model.to(device)
PATH = 'model_epoc_100.pt'
model.load_state_dict(torch.load(PATH,map_location='cpu'))
#%% inference
wav_path='/wavs_16k'
paths=extract_p_paths(wav_path)
import csv
with open('res_T60.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["file_name","T60 predicted","common T60"])

    for path in paths:
        #print(path)
        data = prepare_data(path)
        data = np.asarray(data, dtype='float32')
        data = np.expand_dims(data, axis=0)
        lable_dic = {"0.3":0,"0.45":1,"0.6":2,"0.75":3,"0.9":4,"1.05":5,"1.2":6}

        model.eval()
        with torch.no_grad():
            data = torch.from_numpy(data).to(device)
            # Perform inference
            outputs = model(data.permute(1, 0, 2, 3))

            # Process the outputs
            predicted = torch.max(outputs.data, 1)[1]
            y_pred = np.asarray(predicted.cpu(),dtype='float32')           
            counter = Counter(y_pred.tolist())
            # Find the most common value
            common_value = counter.most_common(1)[0][0]
            T60 = list(lable_dic.keys())[list(lable_dic.values()).index(common_value)]
            writer.writerow([path, y_pred,T60])
            print('T60:', T60)

            
           

