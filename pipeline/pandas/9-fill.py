#!/usr/bin/env python3
'''
This module holds a function
that add some changes to dataframe
'''


def fill(df):
    df.drop(columns=['Weighted_Price'], inplace=True)
    for i in df[df['Close'].isna()].index:
        df.loc[i, 'Close'] = df.loc[i-1, 'Close']
    for i in df[df['High'].isna()].index:
        df.loc[i, 'High'] = df.loc[i, 'Close']
    for i in df[df['Low'].isna()].index:
        df.loc[i, 'Low'] = df.loc[i, 'Close']
    for i in df[df['Open'].isna()].index:
        df.loc[i, 'Open'] = df.loc[i, 'Close']

    df[['Volume_(BTC)', 'Volume_(Currency)']].fillna(0)
