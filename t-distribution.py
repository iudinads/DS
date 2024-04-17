import numpy as np
from scipy import stats
np.random.seed(42)

gen_pop = np.trunc(stats.norm.rvs(loc=50, scale=5, size=1000)) #people's opinion
gen_pop[gen_pop>100]=100
print(f'mean = {gen_pop.mean():.3}')
print(f'std = {gen_pop.std():.3}')


#another situation (opinion about new beer)
other = np.array([60,62,64,63,54,70,90,90,60,85])


#not true distribution
z = 10 ** 0.5 * (other.mean() - 70) / 5
print("z-score is : ", z)

p_value = 1 - (stats.norm.cdf(z) - stats.norm.cdf(-z))
print("p-value is : ", p_value)

print("sigma is : ", other.std(ddof = 1))
print("sigma is too much")

#t-distribution
n = stats.ttest_1samp(other, 50)
# interval
ci = stats.t.interval(0.95, df = 9, loc = other.mean(), scale = other.std(ddof = 1)/10**0.5)
print("interval is : ", ci) #what mark will be given to new beer
