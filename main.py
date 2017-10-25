import datetime
import re
import urllib.request
import dateutil.parser as dparser
import csv
import os

def htmlToData_toCsv(filename):
    file = open(filename, 'r')

    file_str = file.read()
    file_str = file_str.replace('\n','')
    file_str = file_str.replace('\t','')
    file_str = file_str.replace(' ','')

    rows = re.split(r'<tr.*?>(.+?)</tr>', file_str)
    rows = list(filter(None, rows))
    rows = rows[2:-1]

    values = []
    for i in rows:
        data = re.split(r'<td.*?>(.+?)</td>', i)
        data = list(filter(None, data))

        data[0] = dparser.parse(data[0], fuzzy=True)
        year = data[0].strftime('%Y')
        month = data[0].strftime('%m')
        day = data[0].strftime('%d')
        maxi = data[1]
        mini = data[2]

        data = []
        data.append(year)
        data.append(month)
        data.append(day)
        data.append(maxi)
        data.append(mini)

        values.append(data)

    with open("weather_data.csv", "a") as myfile:
        wr = csv.writer(myfile)
        for i in values:
            wr.writerow(i[0:5])


files = os.listdir(os.getcwd())

files_valid = []
for i in files:
    if '20' in i:
        files_valid.append(i)

print(files_valid)

for i in files_valid:
    htmlToData_toCsv(i)
