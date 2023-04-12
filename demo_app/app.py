import pickle
import json 
import gradio as gr
import numpy as np
import pandas as pd
import sklearn
import xgboost
from xgboost import XGBRegressor


# File Paths
model_path = "xgbr_model.sav"#'lgbm_model.sav'
endoing_path = "cat_encods.json"
component_config_path = "component_configs.json"
examples_path = "examples.pkl"

# predefined
target = "Item_Outlet_Sales"

feature_order = ['Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type',
                 'Item_MRP', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type',
                 'New_Item_Type', 'Outlet_Years']

# Loading the files
model = pickle.load(open(model_path, 'rb'))

# loading the classes & type casting the encoding indexes
classes = json.load(open(endoing_path, "r"))
classes = {k:{int(num):cat for num,cat in v.items() } for k,v in classes.items()}

inverse_class = {col:{val:key for key, val in clss.items()}  for col, clss in classes.items()}

#classes[target].values()
examples = pickle.load(open(examples_path, 'rb'))

feature_limitations = json.load(open(component_config_path, "r"))


# Util functions
def decode(col, data):
  return classes[col][data]

def encode(col, str_data):
  return inverse_class[col][str_data]

def feature_decode(df):

  # exclude the target var
  cat_cols = list(classes.keys())
  if "Item_Outlet_Sales" in cat_cols:
    cat_cols.remove("Item_Outlet_Sales")

  for col in cat_cols:
     df[col] = decode(col, df[col])

  return df

def feature_encode(df):
  
  # exclude the target var
  cat_cols = list(classes.keys())
  if "Item_Outlet_Sales" in cat_cols:
    cat_cols.remove("Item_Outlet_Sales")
  
  for col in cat_cols:
     df[col] = encode(col, df[col])
  
  return df


def predict(*args):

  # preparing the input into convenient form
  features = pd.Series([*args], index=feature_order)
  features = feature_encode(features)
  features = np.array(features).reshape(-1,len(feature_order))

  # prediction
  pred = model.predict(features) #.predict(features)


  return np.round(pred, 3)


inputs = list()
for col in feature_order:
  if col in feature_limitations["cat"].keys():
    
    # extracting the params
    vals = feature_limitations["cat"][col]["values"]
    def_val = feature_limitations["cat"][col]["def"]
    
    # creating the component
    inputs.append(gr.inputs.Dropdown(vals, default=def_val, label=col))
  else:
    
    # extracting the params
    min = feature_limitations["num"][col]["min"]
    max = feature_limitations["num"][col]["max"]
    def_val = feature_limitations["num"][col]["def"]
    
    # creating the component
    inputs.append(gr.inputs.Slider(minimum=min, maximum=max, default=def_val, label=col) )


# creating the app
demo_app = gr.Interface(predict, inputs, "number",examples=examples)

# Launching the demo
if __name__ == "__main__":
    demo_app.launch()