import xgboost
from xgboost import XGBRanker, DMatrix
from sklearn.model_selection import GridSearchCV, ParameterGrid
import pandas as pd
from sklearn.metrics import ndcg_score
from functools import partial
import statistics as stats
import csv
import joblib


def train_val_split(data, percentage=0.8):
    n_qids = data['srch_id'].nunique()

    splitpoint = int(n_qids * percentage)
    # print(splitpoint)

    data['unique_id'] = data['srch_id'].rank(method='dense')

    X_train = data[data['unique_id'] < splitpoint].drop(columns=['unique_id'])

    # print(data.shape)
    # print(data['unique_id'])
    # data['unique_id'].to_csv('fu.csv')
    X_test = data[data['unique_id'] >= splitpoint].drop(columns=['unique_id'])
    return X_train, X_test


def makemodel():
    # 5336 entries

    data = pd.read_csv('data/pre_processed_data.csv')

    # print(test_data.columns)
    # print(data.columns)
    # Index(['Unnamed: 0', 'srch_id', 'site_id', 'visitor_location_country_id',
    #        'prop_country_id', 'prop_id', 'prop_starrating', 'prop_review_score',
    #        'prop_brand_bool', 'prop_location_score2', 'prop_log_historical_price',
    #        'promotion_flag', 'srch_destination_id', 'srch_length_of_stay',
    #        'srch_booking_window', 'srch_adults_count', 'srch_children_count',
    #        'srch_room_count', 'srch_saturday_night_bool',
    #        'srch_query_affinity_score', 'orig_destination_distance', 'random_bool',
    #        'year', 'month', 'comp_rate', 'comp_inv', 'comp_rate_percent_diff',
    #        'usd_diff', 'starrating_diff', 'target'],
    #       dtype='object')

    categorical = []

    for i in data.columns:
        # if 'id' in i or 'bool' in i:
        if 'bool' in i:
            categorical.append(i)
    # for col in categorical:
    #    data[col] = data[col].astype('boolean')

    X_train, X_val = train_val_split(data)

    qid_train = X_train['srch_id']
    qid_val = X_val['srch_id']

    y_train = X_train['target']
    y_val = X_val['target']

    print(X_train.columns)
    X_train = X_train.drop(columns=['target', 'srch_id'])
    X_val = X_val.drop(columns=['target', 'srch_id'])

    # print(test_data.columns)
    # X = X.drop(columns = ['target', 'srch_id'])

    # params = {'objective': 'rank:pairwise', 'learning_rate': 0.1,
    #          'gamma': 1.0, 'min_child_weight': 0.1,
    #          'max_depth': 6, 'n_estimators': 4}
    params = {'objective': 'rank:pairwise', 'eval_method': partial(ndcg_score, k=5)}

    ranker = XGBRanker(**params)
    print('Go coffee')

    ranked = ranker.fit(X=X_train, y=y_train, qid=qid_train, eval_set=[(X_val, y_val)], eval_qid=[qid_val])
    print(ranked.evals_result())
    ranked.save_model('trial.json')
    for i, feature in enumerate(ranked.feature_importances_):
        print('importance = ', ranked.feature_importances_[i])
        print('name = ', ranked.feature_names_in_[i])
    return params


def test_and_output(params=None, saved_model='RANKBOOST.json', data='data/test_processed_data.csv',
                    output_name='output_xgboost.csv'):
    test_data = pd.read_csv(data)
    X_test = test_data
    print(X_test.columns)

    if params == None:
        params = {'objective': 'rank:pairwise', 'learning_rate': 0.1,
                  'gamma': 1.0, 'min_child_weight': 0.1,
                  'max_depth': 6, 'n_estimators': 4}
    ranker = XGBRanker(**params)

    ranker.load_model(saved_model)

    predictions = []

    for srch_id in X_test['srch_id'].unique():
        to_predict = X_test[X_test['srch_id'] == srch_id]

        predicted = ranker.predict(to_predict)
        predictions.append(predicted)
        # X_test[to_predict]['predicted'] = predicted
    # predictions = ranked.predict(X_test)
    predictions = [item for sublist in predictions for item in sublist]

    print(predictions)
    X_test['relevance score'] = predictions
    X_test.to_csv('XGBoost_output.csv')

    output = X_test[['srch_id', 'relevance score', 'prop_id']]
    output = output.sort_values(['srch_id', 'relevance score'], ascending=[True, False]).groupby('srch_id').head(25)
    output = output.drop('relevance score', axis=1)
    output.to_csv(output_name, index=False)
    # print(X_test[to_predict])

    # ranked.save_model("RANKBOOST.json")
    # print(ranker.feature_importances_)


def tune():
    data = pd.read_csv('data/pre_processed_data.csv')
    X_train, X_val = train_val_split(data)

    qid_train = X_train['srch_id']
    qid_val = X_val['srch_id']

    y_train = X_train['target']
    y_val = X_val['target']

    print(X_train.columns)
    X_train = X_train.drop(columns=['target', 'srch_id', 'site_id', 'visitor_location_country_id', 'prop_country_id', 'prop_id',
                                    'prop_brand_bool', 'random_bool', 'position'], axis=1)
    X_val = X_val.drop(columns=['target', 'srch_id', 'site_id', 'visitor_location_country_id', 'prop_country_id', 'prop_id',
                                    'prop_brand_bool', 'random_bool', 'position'], axis=1)

    params = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 6, 12],
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 2, 5]
    }
    scores = {}

    paramgrid = ParameterGrid(params)
    print(paramgrid)
    high_score = 0
    for i, param in enumerate(paramgrid):
        if i < 17:
            continue
        print(param)
        xgb = XGBRanker(learning_rate=0.01, objective='rank:pairwise', eval_method=partial(ndcg_score, k=5), **param)
        ranked = xgb.fit(X=X_train, y=y_train, qid=qid_train, eval_set=[(X_val, y_val)], eval_qid=[qid_val])
        score = stats.mean(ranked.evals_result()['validation_0']['map'])
        print(score)
        if score > high_score:

            high_score = score
            ranked.save_model('best_model.json')
            # param.values().to_csv('best_params.csv')
            w = csv.writer(open("best_params1.csv", "w"))

            # loop over dictionary keys and values
            for key, val in param.items():
                # write every key and value to file
                w.writerow([key, val])

        scores[param.values()] = score

    return scores, paramgrid


# params = makemodel()
# params = {'objective':'rank:pairwise', 'eval_method' : partial(ndcg_score, k=5)}
# model = XGBRanker(**params)
# model.load_model('trial.json')
# print(dir(model))
#
# print(model._get_param_names())
# for i, feature in enumerate(model.feature_importances_):
#     print('importance = ', model.feature_importances_[i])
#
# print(model.evals_result())
# print(len(model.evals_result()['validation_0']['map']))
# data = pd.read_csv('data/pre_processed_data.csv')
# X_train, X_val = train_val_split(data)
# print(X_val['srch_id'].nunique())
# print(model.get_params)
#params = {'gamma', 'max_depth', 'min_child_weight', 'n_estimators'}

#with open('best_params1.csv', 'r') as f:
#    csv_reader = csv.DictReader(f)
#    for row in csv_reader:
#        print(row)
#        params[str(row[0])] = row[1]

#print(params)
#test_and_output(saved_model='best_model.json', params=params, output_name='XGtuned.csv')


def evaluation():
    # load model
    params = {'objective': 'rank:pairwise', 'learning_rate': 0.1,
              'gamma': 0.5, 'min_child_weight': 1,
              'max_depth': 12, 'n_estimators': 150}
    ranker = XGBRanker(**params)

    ranker.load_model('RANKBOOST.json')
    # load test data
    test_df = pd.read_csv("data/test_processed_data.csv")
    x_test = test_df.drop(columns=['srch_id', 'site_id', 'visitor_location_country_id', 'prop_country_id', 'prop_id',
                                   'prop_brand_bool', 'random_bool'], axis=1)

    predictions = ranker.predict(x_test)
    d = {'relevance score': predictions}
    prediction_output = pd.DataFrame(d)
    prediction_output['srch_id'] = test_df['srch_id']
    prediction_output['prop_id'] = test_df['prop_id']

    output = prediction_output.sort_values(['srch_id', 'relevance score'], ascending=[True, False]).groupby(
        'srch_id').head(25)
    output = output.drop('relevance score', axis=1)
    output.to_csv('result_xgboost.csv', index=False)


evaluation()