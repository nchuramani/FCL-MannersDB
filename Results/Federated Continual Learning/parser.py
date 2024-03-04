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
                  formatters={"Method": str.upper},
                  float_format="{:.3}".format,caption=caption) , df.iloc[df['PCC2'].idxmax()]
paths = os.listdir()
paths.remove("parser.py")
for path in paths:
  for i in os.listdir(path):
    central=pd.DataFrame(columns=["Method", "reg_coeff", "Loss1", "RMSE1", "PCC1", "Loss2", "RMSE2", "PCC2"])
    decentral=pd.DataFrame(columns=["Method", "reg_coeff", "Loss1", "RMSE1", "PCC1", "Loss2", "RMSE2", "PCC2"])
    #check if path+i is a directory
    if not os.path.isdir(path+"/"+i):
      continue
    for j in os.listdir(path+"/"+i):
      #check if path+i+j is a directory
      if not os.path.isdir(path+"/"+i+"/"+j):
        continue
      for k in os.listdir(path+"/"+i+"/"+j):
        #concat


        for l in os.listdir(path+"/"+i+"/"+j+"/"+k):
          if l[-13:]=='decentral.csv':
            la, dec=latexmaker(path+"/"+i+"/"+j+"/"+k+"/"+l, f"{j}_{k}")
            with open(path+"/"+i+"/"+j+"/"+k+"/"+l[:-4]+".tex","w+") as f:
              f.write(la)
          elif l[-11:]=='central.csv':
            la, cen=latexmaker(path+"/"+i+"/"+j+"/"+k+"/"+l, f"{j}_{k}")
            with open(path+"/"+i+"/"+j+"/"+k+"/"+l[:-4]+".tex","w+") as f:
              f.write(la)
        print("")
        cen['Method']=f"{j}-{cen['Method']}"
        dec['Method']=f"{j}-{dec['Method']}"
        #replace _ in cen['Method'] with - and same in dec
        cen['Method']=cen['Method'].replace("_","-")
        dec['Method']=dec['Method'].replace("_","-")

        central=pd.concat([central,cen.to_frame().T], axis=0)
        decentral=pd.concat([decentral,dec.to_frame().T], axis=0)
    #convert to latex
   
    central.to_csv(f"{path}/{i}/central.csv")
    decentral.to_csv(f"{path}/{i}/decentral.csv")
    central_tex, _=latexmaker(path+"/"+i+"/central.csv", f"{path}_Centralized")
    decentral_tex, _=latexmaker(path+"/"+i+"/decentral.csv", f"{path}_Decentralized")
    with open(path+"/"+i+"/central.tex","w+") as f:
       f.write(central_tex)
    with open(path+"/"+i+"/decentral.tex","w+") as f:
       f.write(decentral_tex)

    print(f'Completed {path}')