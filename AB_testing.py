# I have a list of marks for 2 types of ice creams
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

#read the data
d = pd.read_csv('my_DS/dataA.csv') 
dd = pd.read_csv('my_DS/dataB.csv')

df0 = pd.DataFrame(d)
df2 = pd.DataFrame(dd)

#clean data
print(df0['mark'].isnull().sum())
df1 = df0.dropna()
df1 = df1.reset_index(drop=True)

df1['source'] = 'DataFrame1'
df2['source'] = 'DataFrame2'
combo_df = pd.concat([df1, df2])


# EDA

#1 mean
print(df1['mark'].mean())
print(df2['mark'].mean())

#2 visualization with box and KDE
plt.figure(figsize=(10, 6))
sns.boxplot(x='source',y ='mark',data=combo_df)
plt.title('box with mustache')
plt.show()

sns.histplot(x = 'source', y = 'mark', data = combo_df)
plt.title('распределение оценок с KDE')
plt.show()

plt.figure(figsize=(6, 6))
stats.probplot(combo_df['mark'], dist='norm', plot=plt)
plt.title('Q-Q')    
plt.xlabel('Quantile of normal distibution')
plt.ylabel('QUantile of sample')
plt.grid()
plt.show()

#start A/B testing
#1 data in interquarile range

lower_bound = df1['mark'].quantile(0.25)
upper_bound = df1['mark'].quantile(0.75)

lower_bound1 = df2['mark'].quantile(0.25)
upper_bound1 = df2['mark'].quantile(0.75)

final_df1 = df1[(df1['mark'] >= lower_bound) & (df1['mark'] <= upper_bound)]
final_df2 = df2[(df2['mark'] >= lower_bound) & (df2['mark'] <= upper_bound)]

#using t test // H0 = there no difference
t_statistic, p_value = stats.ttest_ind(final_df1['mark'], final_df2['mark'])
alpha = 0.05

if p_value < alpha:
    print('there is difference')
else:
    print('there is no difference')
