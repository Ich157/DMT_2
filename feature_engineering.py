import numpy as np
import pandas as pd


def divide_time(df):
    # add columns year and month, easier for later feature engineering
    df["date_time"] = pd.to_datetime(df["date_time"])
    df["year"] = df["date_time"].dt.year
    df["month"] = df["date_time"].dt.month

    train_df = df.drop('date_time', 1)

    return train_df
    # TODO: sum up the competetor variables to one variable only


def combine_comp(df):
    df['comp_rate'] = df.fillna(0)['comp1_rate'] + df.fillna(0)['comp2_rate'] + df.fillna(0)['comp3_rate'] + df.fillna(0)['comp4_rate'] + df.fillna(0)['comp5_rate'] + df.fillna(0)[
        'comp6_rate'] + df.fillna(0)['comp7_rate'] + df.fillna(0)['comp8_rate']
    df['comp_inv'] = df.fillna(0)['comp1_inv'] + df.fillna(0)['comp2_inv'] + df.fillna(0)['comp3_inv'] + df.fillna(0)['comp4_inv'] + df.fillna(0)['comp5_inv'] + df.fillna(0)[
        'comp6_inv'] + df.fillna(0)['comp7_inv'] + df.fillna(0)['comp8_inv']
    df['comp_rate_percent_diff'] = df.fillna(0)['comp1_rate_percent_diff'] + df.fillna(0)['comp2_rate_percent_diff'] + df.fillna(0)[
        'comp3_rate_percent_diff'] + df.fillna(0)['comp4_rate_percent_diff'] + df.fillna(0)['comp5_rate_percent_diff'] + df.fillna(0)[
                                       'comp6_rate_percent_diff'] + df.fillna(0)['comp7_rate_percent_diff'] + df.fillna(0)[
                                       'comp8_rate_percent_diff']
    #df['comp_rate'].fillna(0, inplace=True)
    #df['comp_inv'].fillna(0, inpplace=True)
    #df['comp_rate_percent_diff'].fillna(0, inplace=True)
    df = df.drop(
        columns=['comp1_rate', 'comp2_rate', 'comp3_rate', 'comp4_rate', 'comp5_rate', 'comp6_rate', 'comp7_rate',
                 'comp8_rate',
                 'comp1_inv', 'comp2_inv', 'comp3_inv', 'comp4_inv', 'comp5_inv', 'comp6_inv', 'comp7_inv', 'comp8_inv',
                 'comp1_rate_percent_diff', 'comp2_rate_percent_diff', 'comp3_rate_percent_diff',
                 'comp4_rate_percent_diff',
                 'comp5_rate_percent_diff', 'comp6_rate_percent_diff', 'comp7_rate_percent_diff',
                 'comp8_rate_percent_diff'], axis=1)

    return df


def delete_col(df, col_name):
    df = df.drop([col_name], axis=1)
    return df


def fill_col_with_worst(df, col_name, value):
    return df[col_name].fillna(value, inplace=True)


def avg_ranking(df):
    hotels = pd.unique(df['prop_id'].values.ravel())
    means = df.groupby('prop_id')['position'].mean()
    for hotel in hotels:
        df.loc[df['prop_id'] == hotel, 'avg_position'] = means[hotel]

    return df


def is_cheapest(df):
    min_price = df.groupby('srch_id')['price_usd'].min()

    for price in min_price:
        df.loc[df['price_usd'] == price, 'is_cheapest'] = 1
        df['is_cheapest'].fillna(0)

    return df


def starrating_diff(df):
    df['visitor_hist_starrating'].fillna(0, inplace=True)
    df['prop_starrating'].fillna(0, inplace=True)
    df['starrating_diff'] = np.abs(df['visitor_hist_starrating'] - df['prop_starrating'])
    return df


def usd_diff(df):
    df['visitor_hist_adr_usd'].fillna(0, inplace=True)
    df['price_usd'].fillna(0, inplace=True)
    df['usd_diff'] = np.abs(df['visitor_hist_adr_usd'] - df['price_usd'])
    return df


def normalise(df, col_name):
    return df[col_name] / np.nanmax(df[col_name])
