import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import feature_engineering as fe


def prepro(train_df, test_df):
    """
    if pre processing test_data then change all dataframe to test_df expect the first parameter in avg_ranking!!!
    :param train_df:
    :param test_df:
    :return:
    """
    test_df = fe.divide_time(test_df)
    print("time")
    test_df = fe.combine_comp(test_df)
    print("combine comp")
    test_df = fe.avg_ranking(train_df, test_df)
    print("finished avg ranking")
    df = fe.is_cheapest(test_df)
    print("is cheapest")
    df = fe.usd_diff(df)
    df = fe.starrating_diff(df)
    df = df.drop(columns=['visitor_hist_starrating', 'visitor_hist_adr_usd', 'price_usd',
                          'prop_location_score1'], axis=1)

    # fill hotel descritptions with worst possible value
    df['prop_starrating'].fillna(np.amin(df['prop_starrating']), inplace=True)
    df['prop_review_score'].fillna(np.amin(df['prop_review_score']), inplace=True)
    df['prop_location_score2'].fillna(np.amin(df['prop_location_score2']), inplace=True)
    df['srch_query_affinity_score'].fillna(np.amin(df['srch_query_affinity_score']), inplace=True)
    df['orig_destination_distance'].fillna(np.nanmedian(df['orig_destination_distance']), inplace=True)
    print(df.isna().sum())

    return df


def train_preprocessing(df):
    df["click_bool"] = df["click_bool"].astype(int)
    df["booking_bool"] = df["booking_bool"].astype(int)
    df["target"] = df["click_bool"] + 5 * df["booking_bool"]
    df = df.drop(columns=['click_bool', 'booking_bool'], axis=1)

    return df


if __name__ == '__main__':
    train_df = pd.read_csv("data/training_set_VU_DM.csv")
    test_df = pd.read_csv("data/test_set_VU_DM.csv")
    # train_df = prepro(train_df, test_df)
    # train = train_preprocessing(train_df)
    # train.to_csv('data/pre_processed_data.csv', index=False)

    # for test
    # test = pd.read_csv("data/test_set_VU_DM.csv")
    test = prepro(train_df, test_df)
    test.to_csv('data/test_processed_data.csv', index=False)
