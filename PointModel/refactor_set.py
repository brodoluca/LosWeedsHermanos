import pandas as pd
import os

path =  "./MOD2-01/"

all_directories = os.listdir(path)

label_files = [x for x in all_directories if x.endswith(".csv")]

names = ["Image", "X", "Y", "Folder"]


df = pd.DataFrame()
sum = 0
for file in label_files:
    if(file == "" or file == "complete_labels.csv"):
        continue
    temp =  pd.read_csv(path+file,index_col=None)
    sum += len(temp) 
    temp['Folder'] = [file.split(".")[0]]*len(temp["Image"])
    df = pd.concat([df, temp], ignore_index=True)

#df = df.iloc[:-1]
filepath = './complete_labels.csv'
df.to_csv(filepath, index=range(len(df["Image"]))) 
print(len(df), sum)
print(df.tail())

