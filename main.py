import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

def combine_comp(df):
    df['comp_rate'] = df['comp1_rate'] + df['comp2_rate'] + df['comp3_rate'] + df['comp4_rate'] + df['comp5_rate'] + df['comp6_rate'] + df['comp7_rate'] + df['comp8_rate']
    df['comp_inv'] = df['comp1_inv'] + df['comp2_inv'] + df['comp3_inv'] + df['comp4_inv'] + df['comp5_inv'] + df['comp6_inv'] + df['comp7_inv'] + df['comp8_inv']
    df['comp_rate_percent_diff'] = df['comp1_rate_percent_diff'] + df['comp2_rate_percent_diff'] + df['comp3_rate_percent_diff'] + df['comp4_rate_percent_diff'] + df['comp5_rate_percent_diff'] + df['comp6_rate_percent_diff'] + df['comp7_rate_percent_diff'] + df['comp8_rate_percent_diff']
    df['comp_rate'].fillna(0)
    df['comp_inv'].fillna(0)
    df['comp_rate_percent_diff'].fillna(0)
    return df.drop(columns=['comp1_rate','comp2_rate','comp3_rate','comp4_rate','comp5_rate','comp6_rate','comp7_rate','comp8_rate',
                            'comp1_inv','comp2_inv','comp3_inv','comp4_inv','comp5_inv','comp6_inv','comp7_inv','comp8_inv',
                            'comp1_rate_percent_diff','comp2_rate_percent_diff','comp3_rate_percent_diff','comp4_rate_percent_diff',
                            'comp5_rate_percent_diff','comp6_rate_percent_diff','comp7_rate_percent_diff','comp8_rate_percent_diff'])
def preprocessing(df):
    df = combine_comp(df)
    return df

def train_preprocessing(df):
    df["target"] = int(df["click_bool"]) + 3 * int(df["booking_bool"])
    return df.drop(columns=['position','click_bool','booking_bool','gross_booking_usd'])

if __name__ == '__main__':
    train = pd.read_csv("data/training_set_VU_DM.csv")
    test = pd.read_csv("data/test_set_VU_DM.csv")
    train = preprocessing(train)
    test = preprocessing(test)
    train = train_preprocessing(train)
