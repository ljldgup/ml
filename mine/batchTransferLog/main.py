import re, os
import pandas as pd
from datetime import datetime


def readMybatisTime(*files: str):
    insertPattern = re.compile('(\d+)ms.+(INSERT)\s+INTO\s+(\w+)')
    selectPattern = re.compile('(\d+)ms.+(SELECT).+FROM\s+(\w+)')
    updatePattern = re.compile('(\d+)ms.+(UPDATE)\s+(\w+)')
    rst = []
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            line = f.readline()
            while line:
                for pattern in [insertPattern, selectPattern, updatePattern]:
                    groups = pattern.search(line)
                    if groups:
                        rst.append((groups.group(2), groups.group(3), int(groups.group(1))))
                        break
                line = f.readline()

    return rst


def readIcbcTime(*files: str):
    requestPattern = re.compile(r'(2021-09-\d\d \d\d:\d\d:\d\d\.\d\d\d).+工行请求数据.+icbc_url=([^}]+)')
    respondPattern = re.compile(r'(2021-09-\d\d \d\d:\d\d:\d\d\.\d\d\d).+工行响应数据')
    rst = []

    startTime = None
    url = None
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            line = f.readline()
            while line:
                groups = requestPattern.search(line)
                if groups:
                    startTime = groups.group(1)
                    url = groups.group(2)
                else:
                    groups = respondPattern.search(line)
                    if groups and startTime:
                        endTime = groups.group(1)
                        msDelta = getMsDelta(startTime, endTime)
                        rst.append(('icbc', url, msDelta))
                        startTime = None
                line = f.readline()
    return rst


def getMsDelta(startTime: str, endTime: str):
    endTime = datetime.strptime(endTime, "%Y-%m-%d %H:%M:%S.%f")
    startTime = datetime.strptime(startTime, "%Y-%m-%d %H:%M:%S.%f")
    delta = endTime - startTime
    return delta.seconds * 1000 + delta.microseconds // 1000


if __name__ == '__main__':
    sqlData = readMybatisTime(r'C:\Users\kmhqumenglian\Desktop\test.log')
    sqlData = pd.DataFrame(sqlData, columns=['type', 'table', 'ms'])
    icbcData = readIcbcTime(r'C:\Users\kmhqumenglian\Desktop\test.log')
    icbcData = pd.DataFrame(icbcData, columns=['type', 'table', 'ms'])
    total_time = getMsDelta('2021-09-07 20:14:31.636', '2021-09-07 20:56:47.626')
    sqlData.groupby(['table', 'type']).ms.sum()
