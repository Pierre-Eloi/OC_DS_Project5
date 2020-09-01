#! /usr/bin/env python3
# coding: utf-8

""" This module gathers all functions used for data cleaning and feature engineering
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
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
    date_ref = data[r_feature].max()
    # get the Recency
    recency = (data.groupby("customer_unique_id")
                   .order_purchase_timestamp
                   .max()
                   .apply(lambda x: date_ref - x)
                   .apply(lambda x: x / np.timedelta64(1, 'M')))
    recency.fillna(recency.max(), inplace=True)
    if score:
        recency = (recency.apply(lambda x: 10 - x)
                          .round()
                          .where(recency>1, other=1))
    # get the Monetary_value
    monetary = (data.groupby("customer_unique_id")
                    .price
                    .sum()
                    .fillna(0))
    if m_mean:
        monetary /= (data.groupby("customer_unique_id")
                         .order_id
                         .nunique())
    if score:
        monetary = (pd.qcut(monetary, q=10,
                            labels=np.linspace(1, 10, 10))
                      .astype(int))
    # get the Frequency
    frequency = (data.groupby("customer_unique_id")
                     .order_id
                     .nunique()
                     .fillna(0))
    if score:
        frequency = (frequency.where(frequency>0, other=1)
                              .where(frequency<10, other=10))
    # Create the DataFrame
    df = pd.DataFrame({"Recency": recency,
                       "Frequency": frequency,
                       "Monetary_value": monetary},
                      index=recency.index)
    return df

def products_per_order(data):
    """Function to get the average number
    of products per order.
    Parameters:
    data: DataFrame
        the pandas object holding data
    -----------
    Return: Series
        the pandas object holding data
    """
    s = data.groupby("customer_unique_id")["product_id"].count()/ \
        data.groupby("customer_unique_id")["order_id"].nunique()
    return s.fillna(0)

def product_type(data):
    """Function to get the type product for each customer.
    Parameters:
    data: DataFrame
        the pandas object holding data
    -----------
    Return: tuple
        * The list of the features
        * the numpy object holding data
    """
    # create a dataframe will all required features
    features = ["price",
                "product_description_lenght",
                "freight_value"]
    
    df = data.groupby("customer_unique_id")[features].mean()
    # Impute missing data by the mean
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    X = imp_mean.fit_transform(df.values)
    return features, X

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
    review_timedelta = 10 - review_timedelta
    review_timedelta = (review_timedelta.round()
                                        .fillna(0)
                                        .where(review_timedelta>1,
                                               other=1))
    # get the number of review per customer
    review_n = (df.groupby("customer_unique_id")
                  .review_id
                  .count())
    review_n = (review_n.fillna(0)
                        .where(review_n>0, other=1)
                        .where(review_n<10, other=10))
    # get the review mark
    review_mark = (df.groupby("customer_unique_id")
                     .review_score
                     .mean())
    review_mark *=2
    review_mark = (review_mark.fillna(review_mark.mean())
                              .round())
    # get the score
    review = review_timedelta + review_n + review_mark
    review /= 3
    return review.round()

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
    1) removing the mean and scaling to unit variance.
    2) Increasing the weigh of RFM features
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
    std_scaler = StandardScaler()
    X_scaled = std_scaler.fit_transform(X)
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
