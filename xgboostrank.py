import xgboost
from xgboost import XGBRanker, DMatrix
from sklearn.model_selection import cross_val_score
import pandas as pd


def train_val_split(data, percentage = 0.8):
    n_qids = data['srch_id'].nunique()

    splitpoint = int(n_qids * percentage)
    print(splitpoint)

    data['unique_id'] = data['srch_id'].rank(method='dense')

    X_train = data[data['unique_id'] < splitpoint].drop(columns=['unique_id'])

    #print(data.shape)
    #print(data['unique_id'])
    #data['unique_id'].to_csv('fu.csv')
    X_test = data[data['unique_id'] >= splitpoint].drop(columns=['unique_id'])
    return X_train, X_test
def makemodel():
    #5336 entries

    data = pd.read_csv('data/pre_processed_data.csv')

    print(test_data.columns)
    #print(data.columns)
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


    X_train, X_val = train_val_split(data)

    qid_train = X_train['srch_id']
    qid_val = X_val['srch_id']

    y_train = X_train['target']
    y_val = X_val['target']

    print(X_train.columns)
    X_train = X_train.drop(columns = ['target', 'srch_id'])
    X_val = X_val.drop(columns = ['target', 'srch_id'])


    #print(test_data.columns)
    #X = X.drop(columns = ['target', 'srch_id'])




    categorical = []

    for i in X_train.columns:
        if 'id' in i or 'bool' in i:
            categorical.append('c')
        else:
            categorical.append('float')


    # print(categorical)
    # datamatrix = DMatrix(X, label = y, qid = qid,  feature_names= X.columns, feature_types= categorical, enable_categorical= True)
    #
    # print(datamatrix.feature_types)
    #datamatrix.set_group([30])

    params = {'objective': 'rank:pairwise', 'learning_rate': 0.1,
              'gamma': 1.0, 'min_child_weight': 0.1,
              'max_depth': 6, 'n_estimators': 4}

    ranker = XGBRanker(**params)

    ranked = ranker.fit(X = X_train, y = y_train, qid = qid_train, eval_set = [(X_val, y_val)], eval_qid = [qid_val])
    print(ranked.evals_result())


test_data = pd.read_csv('data/test_processed_data.csv')
X_test = test_data


params = {'objective': 'rank:pairwise', 'learning_rate': 0.1,
              'gamma': 1.0, 'min_child_weight': 0.1,
              'max_depth': 6, 'n_estimators': 4}
ranker = XGBRanker(**params)

ranker.load_model('RANKBOOST.json')

predictions = []
for srch_id in X_test['srch_id'].unique():
    to_predict = X_test[X_test['srch_id'] == srch_id]
    predictions.append(ranker.predict(to_predict))
#predictions = ranked.predict(X_test)

print(predictions)



#ranked.save_model("RANKBOOST.json")
#print(ranker.feature_importances_)



