import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
housing = pd.read_csv("housing.csv")

housing.head()
housing.info()

housing["ocean_proximity"].value_counts() #counting the no of categorical values in the dataset

housing["population"].describe()  #Ganing the insights of the dataset


housing.hist(bins=50, figsize=(20,15))  # Histogram analysis of the data features
plt.show()

#Looking for correlations between the features
#Method 1 using the correlation matrix
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
#

#Method 2 using graphs for visuals
from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms",
 "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))

housing.plot(kind = "scatter",y="median_house_value",x="median_income")

#Splitting of the dataset into train and test set
#Method1
from zlib import crc32
def test_set_check(identifier, test_ratio):
 return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32
def split_train_test_by_id(data, test_ratio, id_column):
 ids = data[id_column]
 in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
 return data.loc[~in_test_set], data.loc[in_test_set]

#Split on the basis of index of rows
housing_with_id = housing.reset_index() # adds an `index` column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

#Split on the basis of user created id
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

#Split using slearn library
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
#But these test set created may not represent the entire population so stratificatiion of the test set has to be done 
#so that prediction result is on a dataset which represent almost all category of the population of dataset

#Here stratification can be done on the basis of median income
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
housing["income_cat"].hist()
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
 strat_train_set = housing.loc[train_index]
 strat_test_set = housing.loc[test_index]

strat_test_set["income_cat"].value_counts() / len(strat_test_set) 

#Dropping of the additional attribute used for stratification i.e income_cat
for set_ in (strat_train_set, strat_test_set):
 set_.drop("income_cat", axis=1, inplace=True) 
 
housing_strat = strat_train_set.copy()  ## Creating a copy of the stratified Train set

housing.plot(kind="scatter", x="longitude", y="latitude",alpha = 0.1)

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
 s=housing["population"]/100, label="population", figsize=(10,7),
 c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True, )
plt.legend()




