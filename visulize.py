import datetime
import re
import urllib.request
import dateutil.parser as dparser
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates


df = pd.read_csv("weather_data.csv")
# df["Date"] = pd.to_datetime(df["Date"], format='%m/%d/%Y')
#
# mask0 = (df["Date"] >= '2012-09-01') & (df["Date"] <= '2012-09-30')
# mask1 = (df["Date"] >= '2013-09-01') & (df["Date"] <= '2013-09-30')
# mask2 = (df["Date"] >= '2014-09-01') & (df["Date"] <= '2014-09-30')
# selected1 = df.loc[mask0]
# selected2 = df.loc[mask1]
# selected3 = df.loc[mask2]
#
# # print(selected)
# # print(df["Max"].head())
#
# # dates = matplotlib.dates.date2num(df["Date"])
# print(selected1["Max"].shape)
# print(selected2["Max"].shape)
# print(selected3["Max"].shape)
#
# plt.plot(selected1["Date"], selected1["Max"], "r", selected1["Date"], selected3["Max"], "b")
# plt.show()

print(df.shape)