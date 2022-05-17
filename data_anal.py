import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import csv
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import missingno as msno

train_df = pd.read_csv("data/training_set_VU_DM.csv")
# print(train_df.head)
print(train_df.shape)  # 4958347, 54


# print(train_df["search_id"].value_counts())
# prop_id: 129113 (different hotels), srch_id: 199795 (amount of searches) -> resulting in approx. 25 hotel
# recommendations per search


def plot_nas(df):
    # plot missing data
    msno.bar(df)
    # plot shows that some varibales have missing data


def heatmap(df):
    corr = df.corr()
    sns.heatmap(corr)
    sns.despine()
    plt.tight_layout()
    plt.show()




def missing_interaction(df, booked):
    if booked == 1:
        target = 'booking_bool'
    else:
        target = 'clicked_bool'

    hotel_descriptions = ['prop_starrating', 'prop_review_score', 'prop_location_score2', 'prop_log_historical_price',
                          'srch_query_affinity_score']

    for var in hotel_descriptions:
        booking_made = df[df[target] == 1][var].count()
        no_booking_made = df[df[target] == 0][var].count()

        nan_booked = len(df[df[target] == 1][var]) - booking_made
        nan_non_booked = len(df[df[target] == 0][var]) - no_booking_made

        ratio_booking = []
        ratio_naBooking = []
        ratio_booking.append(float(booking_made) / float(no_booking_made))
        if nan_non_booked == 0:
            ratio_naBooking.append(0)
        else:
            ratio_naBooking.append(float(nan_booked) / float(nan_non_booked))

    # plot data
    ind = np.array(range(len((ratio_booking))))
    width = 0.35
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, ratio_booking, width, color='blue',label='no NA')
    rects2 = ax.bar(ind + width, ratio_naBooking, width, color='green',label='NA')
    ax.set_xticklabels(hotel_descriptions,rotation='vertical')
    plt.legend(prop={'size':6})
    plt.tight_layout()
    plt.show()



plot_nas(train_df)
missing_interaction(train_df, 1)
heatmap(train_df)