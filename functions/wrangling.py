#! /usr/bin/env python3
# coding: utf-8

""" This module gathers all functions used for data cleaning and feature engineering
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def get_rfm(data, score=False, m_mean=False):
    """Function to get the three RFM features,
    commonly used in database marketing.
    Parameters:
    data: DataFrame
        the pandas object holding data
    score: bool, default False
        to get a score between 1 and 10 instead of the true values
    m_mean: bool, default False
        to get the mean and not the total for the monetary value
    -----------
    Return:
        DataFrame
    """
    r_feature = "order_purchase_timestamp"
    f_feature = "order_id"
    m_feature = "price"
    # Set a reference date
    date_max = data[r_feature].max()
    time_delta = np.timedelta64(12, 'M')
    date_ref = date_max - time_delta
    # get the Recency
    recency = (data.groupby("customer_unique_id")
                   .order_purchase_timestamp
                   .max()
                   .apply(lambda x: date_max - x)
                   .apply(lambda x: x / np.timedelta64(1, 'M')))
    recency.fillna(recency.max(), inplace=True)
    if score:
        recency = (recency.apply(lambda x: 5 - x//3)
                          .round())
        recency.where(recency > 1, 1, inplace=True)
    # get the Monetary_value
    if m_mean:
        monetary = (data.groupby("customer_unique_id")
                        .price
                        .sum())
        monetary /= (data.groupby("customer_unique_id")
                         .order_id
                         .nunique())
    else:
        monetary = (data.groupby("customer_unique_id")
                    .price
                    .max())
    monetary.fillna(0, inplace=True)
    if score:
        max_m = monetary.max() + .1 # the rightmost edge is not included
        bins = [0, 50, 100, 500, 1000, max_m]
        monetary = (pd.cut(monetary, bins, right=False,
                           labels=np.linspace(1, 5, 5))
                      .astype(float))
    # get the Frequency
    mask = (data.order_purchase_timestamp >= date_ref)
    frequency = (data[mask].groupby("customer_unique_id")
                           .order_id
                           .nunique()
                           .rename("Frequency"))
    frequency.fillna(0, inplace=True)
    if score:
        frequency = (frequency.apply(lambda x: x + 1)
                              .where(frequency<5, other=5))
    # Create the DataFrame
    df = (pd.DataFrame({"Recency": recency,
                        "Monetary_value": monetary},
                        index=recency.index)
            .merge(frequency, how="outer", left_index=True, right_index=True)
            .loc[:, ["Recency", "Frequency", "Monetary_value"]])
    if score:
        df.fillna(1, inplace=True)
    else:
        df.fillna(0, inplace=True)

    return df

def products_per_order(data, binning=False):
    """Function to get the average number
    of products per order.
    Parameters:
    data: DataFrame
        the pandas object holding data
    binning: bool, default False
        to create a category feature with five categories
    -----------
    Return: Series
        the pandas object holding data
    """
    s = data.groupby("customer_unique_id")["product_id"].count()/ \
        data.groupby("customer_unique_id")["order_id"].nunique()
    s.fillna(0, inplace=True)
    if binning:
        max_n = s.max() + .1 # the rightmost edge is not included
        bins = [0, 1, 3, 5, 10, max_n]
        s = (pd.cut(s, bins, right=False,
                    labels=np.linspace(1, 5, 5))
               .astype(float))
    return s

def product_type(data, binning=False):
    """Function to get the type product for each customer.
    Parameters:
    data: DataFrame
        the pandas object holding data
    binning: bool, default False
        to create a category feature with five categories
    -----------
    Return: tuple
        * The list of the features
        * the numpy object holding data
    """
    # create a dataframe with all required features
    features = ["price",
                "product_description_length",
                "freight_value"]
    df = data.groupby("customer_unique_id")[features].mean()
    mean_desc = df["product_description_length"].mean()
    values = {"price": 0,
              "product_description_length": mean_desc,
              "freight_value": 0}
    df.fillna(values, inplace=True)
    if binning:
        # Price binning
        max_price = df["price"].max() + .1 # the rightmost edge is not included
        bins = [0, 50, 100, 500, 1000, max_price]
        df["price"] = (pd.cut(df["price"], bins, right=False,
                              labels=np.linspace(1, 5, 5))
                         .astype(float))
        # product_description_length binning
        max_length = df["product_description_length"].max() + .1
        bins = [0, 100, 500, 1000, 2000, max_length]
        df["product_description_length"] = (pd.cut(df["product_description_length"],
                                                   bins, right=False,
                                                   labels=np.linspace(1, 5, 5))
                                              .astype(float))
        # freight_value binning
        max_freight = df["freight_value"].max() + .1
        bins = [0, 5, 10, 20, 50, max_freight]
        df["freight_value"] = (pd.cut(df["freight_value"],
                                      bins, right=False,
                                      labels=np.linspace(1, 5, 5))
                                 .astype(float))

    return features, df.values

def review_score(data):
    """Function to get a score of the reviews
    based on three features:
    - the satisfaction survey answer timedelta
    - number of reviews per customer
    - the mark given by the review
    Parameters:
    data: DataFrame
        the pandas object holding data
    -----------
    Return: Series
        the pandas object holding data
    """
    df = data.copy()
    # get the survey answer timedelta in days
    df["review_answer_timedelta"] = df["review_answer_timestamp"]
    df["review_answer_timedelta"] -= df["review_creation_date"]
    df["review_answer_timedelta"] /= np.timedelta64(1, 'D')
    review_timedelta = (df.groupby("customer_unique_id")
                          .review_answer_timedelta
                          .mean())
    review_timedelta = 5 - review_timedelta
    review_timedelta = (review_timedelta.round()
                                        .fillna(0)
                                        .where(review_timedelta>1,
                                               other=1))
    # get the average number of review per order
    review_n = (df.groupby("customer_unique_id")
                  .review_id
                  .count())
    review_n /= (df.groupby("customer_unique_id")
                   .order_id
                   .nunique())
    max_review_n = review_n.max()
    bins = [0, 1, 2, 3, 4, max_review_n]
    review_n = (pd.cut(review_n, bins,
                       labels=np.linspace(1, 5, 5))
                  .astype(float))
    # get the review mark
    review_mark = (df.groupby("customer_unique_id")
                     .review_score
                     .mean())
    review_mark = (review_mark.fillna(review_mark.mean())
                              .round())
    # get the score
    review = review_timedelta + review_n + review_mark
    return review.round(1)

def payment_type(data):
    """Function to get the type of payment used by each customer.
    Parameters:
    data: DataFrame
        the pandas object holding data
    -----------
    Return: tuple
        * The list of the payment_types
        * the numpy object holding data
    """
    df = data.copy()
    df.payment_type.where(df.payment_type.isin(["credit_card", "boleto"]),
                          "other_payment", inplace=True)
    # Create a feature by type of payment
    list_payment_type = (df.payment_type
                           .unique()
                           .tolist())
    for c in list_payment_type:
        cond = df["payment_type"]==c
        df[c] = df["payment_value"]
        df[c] = df[c].where(cond, 0)
    # Group all data by customer
    df = df.groupby("customer_unique_id")[list_payment_type].sum()
    # Calculate the proportions
    for c in list_payment_type:
        df[c] /= (data.groupby("customer_unique_id")
                      .payment_value
                      .sum())
    return list_payment_type, df.fillna(0).values

def scale_data(data):
    """Function to scale data by:
    1) using a min-max scaling
    2) Increasing the weigh of RFM features, thus the ladder will represent 50% of the total weight.
    -----------
    Parameters:
    data: DataFrame
        the pandas object holding data
    -----------
    Return:
        DataFrame
    """
    rfm_feat = ["Recency", "Frequency", "Monetary_value"]
    df = data.copy()
    n = data.columns.size
    weight = (n-3) / 3
    X = data.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    df.loc[:, :] = X_scaled
    df.loc[:, rfm_feat] *= weight
    return df

def data_filter(data, ref_date):
    """Function to filter data by keeping only
    the orders purchased prior to a reference data
    entered in parameter.
    Parameters:
    data: DataFrame
        the pandas object holding data
    ref_date: datetime object
        the reference date to be used to filter data
    -----------
    Return:
        DataFrame
    """
    return data[data.order_purchase_timestamp<=ref_date]

def wrangling_pipeline(data, ref_date=None, m_mean=False, score=False):
    """pipeline to carry out all functions of data wrangling.
    Parameters:
    data: DataFrame
        the pandas object holding data
    ref_date: datetime object
        the reference date to be used to filter data
    m_mean: bool, default False
        to get the mean and not the total for the monetary value
    score: bool, default False
        to get a score between 1 and 10 instead of the true values
    -----------
    Return:
        DataFrame
    """
    if not ref_date:
        ref_date = data.order_purchase_timestamp.max()
    data = data_filter(data, ref_date=ref_date)
    df = get_rfm(data, m_mean=m_mean, score=score)
    df["order_n_products"] = products_per_order(data)
    list_features, product_array = product_type(data)
    for i, c in enumerate(list_features):
        df[c] = product_array[:, i]
    df["review"] = review_score(data)
    list_payment, payment_array = payment_type(data)
    for i, c in enumerate(list_payment):
        df[c] = payment_array[:, i]
    return scale_data(df)
