import numpy as np
import pandas as pd
import json
import xgboost
from xgboost import XGBRanker, DMatrix

model = XGBRanker()
model.load_model("best_model.json")

test = pd.read_csv("data/test_set_VU_DM.csv")
print(test)