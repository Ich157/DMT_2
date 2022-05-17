import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


# read data
df = pd.read_csv("data/pre_processed_data.csv")
df_test = pd.read_csv("data/test_processed_data.csv")
print(df_test.head)


# separate feature and target variable
x_train = df.drop('target', axis=1)  # Feautures
y_train = df['target']  # Target
x_train = x_train.drop(columns=['srch_id', 'site_id', 'visitor_location_country_id', 'prop_country_id', 'prop_id',
                      'prop_brand_bool','random_bool'], axis=1)
x_test = df_test.drop(columns=['srch_id', 'site_id', 'visitor_location_country_id', 'prop_country_id', 'prop_id',
                      'prop_brand_bool','random_bool'], axis=1)

# fit model
model = RandomForestRegressor(n_estimators=10, random_state=0)
model.fit(x_train, y_train)

# predict
y_pred = model.predict(x_test)
print(y_pred)
print(y_pred.shape)

prediction_output = pd.DataFrame(y_pred)
prediction_output['srch_id'] = df_test['srch_id']
prediction_output['prop_id'] = df_test['prop_id']

prediction_output.to_csv('output.csv', index=False)