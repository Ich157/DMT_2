import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import ndcg_score
import joblib


def read_data():
    # read data
    df_train = pd.read_csv("data/pre_processed_data.csv")
    df_test = pd.read_csv("data/test_processed_data.csv")
    print(df_test.head)
    return df_train, df_test


def data_split(df):
    # get 40000 search_ids for validation set
    search_id = pd.unique(df['srch_id'].values.ravel())
    val_searchids = search_id[:40000]
    # initialize validation set
    val_df = df.loc[df['srch_id'].isin(val_searchids)]
    train_df = df.loc[~df['srch_id'].isin(val_searchids)]

    # separate feature and target variable
    x_train = train_df.drop('target', axis=1)  # Features
    x_train = x_train.drop(columns=['srch_id', 'site_id', 'visitor_location_country_id', 'prop_country_id', 'prop_id',
                                    'prop_brand_bool', 'random_bool', 'position'], axis=1)
    y_train = train_df['target']  # Target

    x_val = val_df.drop('target', axis=1)
    x_val = x_val.drop(columns=['srch_id', 'site_id', 'visitor_location_country_id', 'prop_country_id', 'prop_id',
                                'prop_brand_bool', 'random_bool', 'position'], axis=1)
    y_val = val_df['target']

    return x_train, y_train, x_val, y_val


def fit_model(x_train, y_train, x_val, val_df):
    # fit model
    model = RandomForestRegressor()
    model.fit(x_train, y_train)

    # predict
    y_val_pred = model.predict(x_val)
    print(y_val_pred)
    print(y_val_pred.shape)

    d = {'relevance score': y_val_pred}
    prediction_output = pd.DataFrame(d)
    prediction_output['srch_id'] = val_df['srch_id']
    prediction_output['prop_id'] = val_df['prop_id']
    prediction_output['real_ranking'] = val_df['position']

    val_output = prediction_output.sort_values(['srch_id', '0'], ascending=[True, False]).groupby('srch_id').head(25)
    val_output.to_csv('val_output.csv', index=False)

    ndcg = []
    for x in val_output['srch_id']:
        true_relevance = val_output.loc[val_output['srch_id'] == x, 'position']
        scores = val_output.loc[val_output['srch_id'] == x, 'relevance score']
        ndcg.append(ndcg_score(true_relevance, scores))

    ndcg_mean = np.mean(ndcg)
    print(ndcg_mean)

    return ndcg_mean


def fine_tuning(train_df):
    ### DATASET FOR TUNING
    # make set for fine tuning smaller
    search_id = pd.unique(train_df['srch_id'].values.ravel())
    search_ids = search_id[:40000]
    tuning_data = train_df.loc[train_df['srch_id'].isin(search_ids)]

    # seperate features and target
    x_tuning = tuning_data.drop('target', axis=1)  # Features
    x_tuning_data = x_tuning.drop(columns=['srch_id', 'site_id', 'visitor_location_country_id', 'prop_country_id', 'prop_id',
                                    'prop_brand_bool', 'random_bool', 'position'], axis=1)
    y_tuning = tuning_data['target']

    # DATASET FOR TRAINING
    x_train = train_df.drop('target', axis=1)
    x_train = x_train.drop(columns=['srch_id', 'site_id', 'visitor_location_country_id', 'prop_country_id', 'prop_id',
                                    'prop_brand_bool', 'random_bool', 'position'], axis=1)
    y_train = train_df['target']

    # FINE TUNING WITH RANDOM SEARCH
    # number of trees in random forest
    n_estimators = [100, 200, 300, 400, 1000]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [70, 80, 90, 100, 110, None]
    # Minimum number of samples required to split a node
    min_samples_split = [5, 10, 12]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [2, 4, 8]
    # Method of selecting samples for training each tree
    bootstrap = [True]

    # random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    rf = RandomForestRegressor()
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=20, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)

    # fit random search model
    rf_random.fit(x_tuning_data, y_tuning)

    # print parameters
    print(rf_random.best_params_)

    tuned_model = rf.best_estimator_
    #joblib.dump(model, 'random_forest.pkl')

    # train model with best fine tuned model
    trained_model = tuned_model.fit(x_train, y_train)

    return trained_model


def evaluation(test_df, model):
    # load model
    x_test = test_df.drop(columns=['srch_id', 'site_id', 'visitor_location_country_id', 'prop_country_id', 'prop_id',
                                   'prop_brand_bool', 'random_bool'], axis=1)

    predictions = model.predict(x_test)
    d = {'relevance score': predictions}
    prediction_output = pd.DataFrame(d)
    prediction_output['srch_id'] = test_df['srch_id']
    prediction_output['prop_id'] = test_df['prop_id']

    output = prediction_output.sort_values(['srch_id', 'relevance score'], ascending=[True, False]).groupby(
        'srch_id').head(25)
    output = output.drop('relevance score', axis=1)
    output.to_csv('output.csv', index=False)


def simple_forest(train_df):
    # DATASET FOR TRAINING
    x_train = train_df.drop('target', axis=1)
    x_train = x_train.drop(columns=['srch_id', 'site_id', 'visitor_location_country_id', 'prop_country_id', 'prop_id',
                                    'prop_brand_bool', 'random_bool', 'position'], axis=1)
    y_train = train_df['target']

    rf = RandomForestRegressor(n_estimators=10, random_state=0)
    simple_rf = rf.fit(x_train, y_train)

    return simple_rf


train_df, test_df = read_data()
# x_train, y_train, x_val, y_val = data_split(train_df)
# model = fine_tuning(train_df)
model = simple_forest(train_df)
evaluation(test_df, model)
