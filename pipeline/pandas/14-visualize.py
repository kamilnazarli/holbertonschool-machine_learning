#!/usr/bin/env python3
'''
This module visaulize
pandas dataframe
'''


df.drop(columns=['Weighted_Price'], inplace=True)
df.rename(columns={'Timestamp': 'Date'}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'], unit='s').dt.normalize()
for i in df[df['Close'].isna()].index:
    df.loc[i, 'Close'] = df.loc[i-1, 'Close']

for i in df[df['High'].isna()].index:
    df.loc[i, 'High'] = df.loc[i, 'Close']

for i in df[df['Low'].isna()].index:
    df.loc[i, 'Low'] = df.loc[i, 'Close']

for i in df[df['Open'].isna()].index:
    df.loc[i, 'Open'] = df.loc[i, 'Close']
df[['Volume_(BTC)', 'Volume_(Currency)']] = \
    df[['Volume_(BTC)', 'Volume_(Currency)']].fillna(0)
df.set_index(keys=['Date'], inplace=True)
df = df.loc[df['Date'] >= '2017-01-01']
df.groupby(['Date']).agg(
    High=('High', 'max'),
    Low=('Low', 'min'),
    Open=('Open', 'mean'),
    Close=('Close', 'mean'),
    **{'Volume_(BTC)': ('Volume_(BTC)', 'sum')},
    **{'Volume_(Currency)': ('Volume_(Currency)', 'sum')}

)
df.plot()
