import os
import pandas as pd
def latexmaker(path, caption):
  df=pd.read_csv(path)
  try:
    df.drop(["Unnamed: 0"], inplace=True, axis=1)
  except:
    print("")

  #replace _ in caption with -
  caption=caption.replace("_","-")
  return df.to_latex(index=False,
                  float_format="{:.03f}".format,caption=caption) 
paths=os.listdir()
print(paths)
for path in paths:
    if path[-4:]=='.csv' and "sorted" not in path:
        print("Processing: ",path)
        df=pd.read_csv(path)
        df.drop(["Unnamed: 0"], inplace=True, axis=1)
        aug=pd.DataFrame(columns=df.columns)
        non=pd.DataFrame(columns=df.columns)
        for i in range(df.shape[0]):
            curr=df.iloc[i,:]
            curr['methods']=curr['methods'].replace("_","-")
            if 'Aug' in df['methods'][i]:
                aug=pd.concat([aug,curr.to_frame().T],axis=0)
            else:
                non=pd.concat([non,curr.to_frame().T],axis=0)
        sorted = pd.concat([non,aug])
        sorted.to_csv(path[:-4]+"_sorted.csv")
        la=latexmaker(path[:-4]+"_sorted.csv", f"{path[:-4]}")
        with open(path[:-4]+"_sorted.tex","w+") as f:
            f.write(la)
        
            

