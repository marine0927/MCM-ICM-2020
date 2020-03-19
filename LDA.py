from nltk.tokenize import TweetTokenizer
from nltk.stem.porter import PorterStemmer
import re
import pandas as pd
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
import numpy as np
import matplotlib.pyplot as plt

'''
problem a tf-idf LogisticRegression
'''

stops = stopwords.words("english")

sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='gb18030')

def tokenizer_porter(text):
    tokens = TweetTokenizer().tokenize(text)
    porter = PorterStemmer()
    #使用空格进行分词，提取分词后的单词原型
    return [porter.stem(word) for word in tokens]

def remove_stopwords(text):

    return [word for word in text if word not in stops]

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

def get_X_Y():
    data1 = pd.read_csv("./Resource/pacifier.tsv", sep='\t', header=0)
    data2 = pd.read_csv("./Resource/microwave.tsv", sep='\t', header=0)
    data3 = pd.read_csv("./Resource/hair_dryer.tsv", sep='\t', header=0)
    #获取X去除HTML标记
    X1 = data1["review_body"].apply(remove_HTML_punctuation)
    X2 = data2["review_body"].apply(remove_HTML_punctuation)
    X3 = data3["review_body"].apply(remove_HTML_punctuation)
    X=pd.concat([X1,X2,X3],ignore_index=True)
    #获取Y
    Y1 = data1["star_rating"]
    Y2 = data2["star_rating"]
    Y3 = data3["star_rating"]
    Y=pd.concat([Y1,Y2,Y3],ignore_index=True)


    return X,Y



if __name__ == '__main__':

    # tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
    # # 设置网格参数
    # '''
    # vect__ngram_range:设置元组的数目
    # vect__stop_words:停用词
    # vect__tokenizer:提取词干，获取词的原型
    # clf__penalty:设置logistic回归正则化的方式
    # clf__C:设置正则化系数
    # '''
    # param_grid = [{"vect__ngram_range": [(1, 1)],
    #                "vect__stop_words": [stops, None],
    #                "vect__tokenizer": [tokenizer_porter, None],
    #                "clf__penalty": ["l1", "l2"],
    #                "clf__C": [0.1, 1.0, 10.0, 100.0]
    #                },
    #               {"vect__ngram_range": [(1, 1)],
    #                "vect__stop_words": [stops, None],
    #                "vect__tokenizer": [tokenizer_porter, None],
    #                "vect__use_idf": [False],
    #                "vect__norm": [None],
    #                "clf__penalty": ['l1', 'l2'],
    #                "clf__C": [0.1, 1.0, 10.0, 100.0]}]
    # lr_tfidf = Pipeline([("vect", tfidf), ("clf", LogisticRegression(random_state=0, multi_class="auto",max_iter=3000,solver='liblinear'))])
    # # 5折交叉验证
    # gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring="accuracy", cv=5, verbose=1, n_jobs=-1)
    # X, Y = get_X_Y()
    # # 分割数据，20%作为测试集，80%作为训练集
    # train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.07, random_state=0)
    # # 训练模型
    # gs_lr_tfidf.fit(train_x, train_y)
    # # 获取网格搜索的最好参数
    # print("Best parameter set:%s" % gs_lr_tfidf.best_params_)
    # # 获取5折交叉验证在训练集上的准确率
    # print("Best score:%.3f" % gs_lr_tfidf.best_score_)
    # # 获取模型在测试集上的准确率
    # print("Test Accuracy:%.3f" % gs_lr_tfidf.best_estimator_.score(test_x, test_y))
    #
    #
    # #保存停用词
    # pickle.dump(stops,open("stopwords.pkl","wb"),protocol=4)
    # #保存模型
    # pickle.dump(gs_lr_tfidf.best_estimator_,open("classifier.pkl","wb"),protocol=4)

    '''
    problem e
    '''

    data1 = pd.read_csv("./Resource/pacifier.tsv", sep='\t', header=0)
    data2 = pd.read_csv("./Resource/microwave.tsv", sep='\t', header=0)
    data3 = pd.read_csv("./Resource/hair_dryer.tsv", sep='\t', header=0)
    # 获取X去除HTML标记
    X1 = data1["review_body"].apply(remove_HTML_punctuation)
    X2 = data2["review_body"].apply(remove_HTML_punctuation)
    X3 = data3["review_body"].apply(remove_HTML_punctuation)
    X = pd.concat([X1, X2, X3], ignore_index=True)
    # 获取Y
    Y1 = data1["star_rating"]
    Y2 = data2["star_rating"]
    Y3 = data3["star_rating"]
    Y = pd.concat([Y1, Y2, Y3], ignore_index=True)
    clf=joblib.load("./classifier.pkl")
    list=clf.predict(X)
    print(list)
    neg=[]
    with open('./负面评价词语（英文）.txt', 'r') as f:
        for line in f:
            line=line.split('\n')[0]
            neg.append(line)
    f.close()

    pos=[]
    with open('./正面评价词语（英文）.txt', 'r') as f:
        for line in f:
            line=line.split('\n')[0]
            pos.append(line)
    f.close()

    countpos=[]
    countneg=[]
    count = []
    with open('e adjective.txt', 'w') as f:
        for i in range(32024):
            sumpos = 0
            sumneg = 0
            for word in pos:
                sumpos = sumpos + str(X[i]).count(word)
            for word in neg:
                sumneg = sumneg + str(X[i]).count(word)
            print(i)
            countpos.append(sumpos)
            countneg.append(sumneg)
            count.append((sumpos - sumneg) / len(str(X[i])))
            f.write(str((sumpos - sumneg) / len(str(X[i])))+' '+str(list[i])+'\n')
    f.close()
    # fig = plt.figure()
    # # 将画图窗口分成1行1列，选择第一块区域作子图
    # ax = fig.add_subplot(1, 1, 1)
    # ax.scatter(count, list, c='k', marker='.')
    # plt.show()
    count = pd.Series(count)
    list = pd.Series(list)

    x = np.vstack((list, count))
    r = np.corrcoef(x)
    print(r)




'''
LDA difference value count
'''




    #for i in range(32024):
    #         f.write(str(list[i])+ ' '+str(Y[i])+' '+ str(X[i])+'\n')
    # f.close()
    # x=[0,0,0,0,0,0]
    # for i in range(32024):
    #     a=abs(Y[i]-list[i])
    #     x[a]=x[a]+1
    # print(x)
