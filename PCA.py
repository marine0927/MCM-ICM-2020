# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from datetime import datetime
import math,re
from nltk.tokenize import TweetTokenizer
from nltk.stem.porter import PorterStemmer
import joblib
import math
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from nltk.corpus import stopwords
import sys,io
'''
problem 1
'''

def tokenizer_porter(text):
    tokens = TweetTokenizer().tokenize(text)
    porter = PorterStemmer()
    #使用空格进行分词，提取分词后的单词原型
    return [porter.stem(word) for word in tokens]

def remove_HTML_punctuation(text):
    if type(text).__name__ != 'float':
        text = re.sub('<[^>]*>', '', text)
        # 寻找表情符号
        emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
        # 删除标点符号
        text = re.sub('[\W]+', ' ', text.lower()) + ''.join(emoticons).replace("-", "")
    else:
        text=' '
    return text

def f(t):
    return 1-1/(1+9*math.exp(-0.007*t))

def g(tn):
    return 1.1/(1+3*math.exp(-0.1*(tn/2-math.log(3,10)/0.1)))-0.1

def date2t(date1,date2):

    date1 = date1.split('/')
    day1 = datetime(int(date1[2]), int(date1[0]), int(date1[1]))
    date2 = date2.split('/')
    day2 = datetime(int(date2[2]), int(date2[0]), int(date2[1]))
    t = (day1 - day2).days
    return t



if __name__ == '__main__':

    data1 = pd.read_csv("./Resource/pacifier.tsv", sep='\t', header=0)
    data2 = pd.read_csv("./Resource/microwave.tsv", sep='\t', header=0)
    data3 = pd.read_csv("./Resource/hair_dryer.tsv", sep='\t', header=0)

    vine1 = data1["vine"]
    vine2 = data2["vine"]
    vine3 = data3["vine"]
    vine = pd.concat([vine1, vine2, vine3], ignore_index=True)

    verified_purchase1 = data1["verified_purchase"]
    verified_purchase2 = data2["verified_purchase"]
    verified_purchase3 = data3["verified_purchase"]
    verified_purchase = pd.concat([verified_purchase1, verified_purchase2, verified_purchase3], ignore_index=True)

    helpful_votes1 = data1["helpful_votes"]
    helpful_votes2 = data2["helpful_votes"]
    helpful_votes3 = data3["helpful_votes"]
    helpful_votes = pd.concat([helpful_votes1, helpful_votes2, helpful_votes3], ignore_index=True)

    total_votes1 = data1["total_votes"]
    total_votes2 = data2["total_votes"]
    total_votes3 = data3["total_votes"]
    total_votes = pd.concat([total_votes1, total_votes2, total_votes3], ignore_index=True)

    X = []
    for i in range(32024):
        if vine[i] == 'N' or vine[i] == 'n':
            vine[i] = 0
        elif vine[i] == 'Y' or vine[i] == 'y':
            vine[i] = 1

        if verified_purchase[i] == 'N' or verified_purchase[i] == 'n':
            verified_purchase[i] = 0
        elif verified_purchase[i] == 'Y' or verified_purchase[i] == 'y':
            verified_purchase[i] = 1
        X.append([vine[i], verified_purchase[i], helpful_votes[i], total_votes[i]])

    X = np.array(X)
    pca = PCA(n_components=4)  # 降到2维
    pca.fit(X)  # 训练
    newX = pca.fit_transform(X)  # 降维后的数据
    # PCA(copy=True, n_components=2, whiten=False)
    print(pca.explained_variance_)  # 输出方差
    print(pca.explained_variance_ratio_)  # 输出支持率

    '''
    pca image trend
    '''

    # print(newX)
    # 创建画图窗口
    # sum0=[]
    # sum1=[]
    # for i in range(32024):
    #     if verified_purchase[i]==0:
    #         sum0.append(newX[i][0])
    #     else:
    #         sum1.append(newX[i][0])
    #
    # sum0=np.array(sum0)
    # sum1=np.array(sum1)
    # print(sum0.mean(),sum1.mean())
    # fig = plt.figure()
    # # 将画图窗口分成1行1列，选择第一块区域作子图
    # ax = fig.add_subplot(1, 1, 1)
    # ax.scatter(x, y, c='k', marker='.')
    # plt.show()
    # joblib.dump(pca,'PCA.model')

    '''
    problem d
    '''

    P = []
    for i in range(32024):
        P.append(newX[i][0])
    P = np.array(P)
    for i in range(32024):
        P[i] = (P[i] - P.min()) / (P.max() - P.min())


    review_date = data3["review_date"]

    first_day = '3/2/2002'

    review_body = data3["review_body"].apply(remove_HTML_punctuation)

    clf = joblib.load("./classifier.pkl")
    review_body = clf.predict(review_body)

    for i in range(11470):
        review_date[i]=date2t(review_date[i],first_day)
    print(review_date)


    t=[]
    fame=[]
    tns=[]
    for i in range(4931):
        t.append(i)
        sum = 0
        n=0
        tn=0
        for j in range(11470):

            if  review_date[j]>=i:
                if review_date[j]<i+7:
                    tn=tn+1
                    # print(review_body[j], P[j], f(i - review_date[j]), i - review_date[j])
                    n = n + 1
                    sum = sum + float(review_body[j]) * float(P[j+18939+1615]) * f(i - review_date[j])
        if n > 0:
            fame.append(float(sum / n * g(tn)))
            print(i, tn, g(tn), sum / n * g(tn))
        else:
            fame.append(float(sum))
            print(i, tn, g(tn), sum)


        tns.append(tn)

    score=[]
    with open('./pacifier.txt', 'r') as f:
        for line in f:
            line = line.split('\n')[0]
            score.append(float(line.split(' ')[3]))
    f.close()

    fame=pd.Series(fame)
    score=pd.Series(score)
    x = np.vstack((fame,score))
    r=np.corrcoef(x)
    print(r)



'''
problem b
'''
    # t = []
    # fame = []
    # tns=[]
    #
    # for i in range(4931):
    #     t.append(i)
    #     sum = 0
    #     n=0
    #     tn=0
    #     for j in range(11470):
    #
    #         if  review_date[j]<=i:
    #             if review_date[j]>i-300:
    #                 tn=tn+1
    #             # print(review_body[j], P[j], f(i - review_date[j]), i - review_date[j])
    #             n=n+1
    #             sum = sum + float(review_body[j]) * float(P[j+18939+1615]) * f(i-review_date[j])
    #     if n>0:
    #         fame.append(sum/n*g(tn))
    #         print(i, tn, g(tn), sum / n * g(tn))
    #     else :
    #         fame.append(sum)
    #         print(i, tn, g(tn), sum)
    #     tns.append(tn)
    #
    #
    #
    # plt.plot(t,fame)
    # plt.xlim(0, 4931)
    # plt.show()

