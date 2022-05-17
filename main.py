import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import feature_engineering as fe


def prepro(df):
    df = fe.divide_time(df)
    df = fe.combine_comp(df)

    df = fe.usd_diff(df)
    df = fe.starrating_diff(df)
    df = df.drop(columns=['visitor_hist_starrating', 'visitor_hist_adr_usd', 'price_usd', 'gross_bookings_usd', 'prop_location_score1'], axis=1)

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
    df = df.drop(columns=['position', 'click_bool', 'booking_bool'], axis=1)

    return df


if __name__ == '__main__':
    train_df = pd.read_csv("data/training_set_VU_DM.csv")
    #test = pd.read_csv("data/test_set_VU_DM.csv")
    train_df = prepro(train_df)
    train = train_preprocessing(train_df)
    #test = preprocessing(test)
    train_df.to_csv('data/pre_processed_data.csv', index=False)

