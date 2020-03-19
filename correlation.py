'''
problem d
'''

# import numpy as np
# import pandas as pd
# score =[]
# with open('./microwave.txt', 'r') as f:
#     for line in f:
#         line = line.split('\n')[0]
#         score.append(float(line.split(' ')[3]))
# f.close()
#
# fame7=[]
# with open('./microwave7.txt', 'r') as f:
#     for line in f:
#         line = line.split('\n')[0]
#         fame7.append(float(line.split(' ')[3]))
# f.close()
#
# fame7 =pd.Series(fame7)
# score =pd.Series(score)
#
# x = np.vstack((fame7 ,score))
# r= np.corrcoef(x)
# print(r)

'''
count product type
'''

import pandas as pd
from collections import Counter
import sys,io
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='gb18030')
#
# c=Counter();
# data = pd.read_csv("./Resource/pacifier.tsv", sep='\t', header=0)
# product_title=data["product_title"]
# for product in product_title:
#     c[product]=c[product]+1
# n=0
# for i in c:
#     n=n+1
# print(n)

'''
e average
'''

c=Counter()
n=Counter()
with open('./e emotion.txt', 'r') as f:
    for line in f:
        line=line.split('\n')[0]
        c[line.split(' ')[1]]=c[line.split(' ')[1]]+float(line.split(' ')[0])
        n[line.split(' ')[1]] = n[line.split(' ')[1]] + 1

print(c,n)
for i in range(1,6):
    print(i,c[str(i)]/n[str(i)])