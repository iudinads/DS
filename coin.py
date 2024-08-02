# to check is coin balanced

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

data = pd.read_csv('my_DS/coin.csv')


data['result'].plot(kind='hist',bins = 2, color='pink', alpha=0.7) 
plt.title('Histogram')
plt.xlabel('0 - hrads, 1 - tails')
plt.ylabel('frequency')

plt.xticks([0,1]) 
#plt.show()

results = data['result'].values
heads = np.sum(results == 0)
tails = np.sum(results == 1)
total = heads + tails


#H0 data is normally distributed

statistic, p_value = stats.shapiro(data['result'])
alpha = 0.05

if p_value > alpha:
    print('H0 is True, data is normally distributed')
else:
    print('H0 is False, data is not normally distributed')

# data isn't normally distribured, so we can't use z-test
# use bootstrap


#H0 coin is balanced
observed_proportion = heads / total
print('observed_proportion = ', observed_proportion)

n_iterations = 10000
bootstrapped_proportions = []
for _ in range(n_iterations):
    sample = np.random.choice(results, size=total, replace=True)
    bootstrapped_proportion = np.mean(sample == 0)

bootstrapped_proportions.append(bootstrapped_proportion)
bootstrapped_proportions = np.array(bootstrapped_proportions)

p_value1 = np.mean(np.abs(bootstrapped_proportions - 0.5) >= np.abs(observed_proportion - 0.5))
print('p-значение:', p_value1)

if p_value1 > alpha:
    print('H0 is True, coin is balanced')
else:
    print('H0 is False, coin is not balanced')
