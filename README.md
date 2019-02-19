
# PredictAPrice

## As a property seller in Manhattan, what are factors that affect what you should price at?

### Datasets

We used NYC.gov's Rolling Sales data from the last 12 months, and 2 Zillow datasets: one showing all median prices and one of all square footage of properties sold. Both are in the year 2018. We merged all datasets together into a main dataframe.

### Data Cleaning

We filtered out all 0 value rows and extreme outliers.

### Transforming Data

When we plotted out our graphs, the normal distribution was postively skewed. We log transformed the data and the graph read much better.


### Checking Features for Usability

![Image of Yaktocat](https://octodex.github.com/images/yaktocat.png)

## Import all libraries


```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics

from scipy import stats
from scipy.stats import skew,norm
from scipy.stats.stats import pearsonr


from sklearn.linear_model import LinearRegression

import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats
import statsmodels.stats.api as sms
import seaborn as sns


import matplotlib


# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn import metrics
```

## Access and Filter necessary info from 3 Real Estate datasets


```python
#combined Rolling Sales Manhattan excel sheets for years 2016-2018
data=pd.read_excel("rollingsales_manhattan.xls", skiprows=1) 

##Filter 0 values and very extreme outliers 
data = data[data['SALE_PRICE'] > 100] 
data = data[data['SALE_PRICE'] < 250000000]  
data=data[data['GROSS SQUARE FEET'] > 100] 

#List of Columns pre-filtering
data.columns=['BOROUGH', 'NEIGHBORHOOD', 'BUILDING_CLASS_CATEGORY',
       'TAX_CLASS_AT_PRESENT', 'BLOCK', 'LOT', 'EASE-MENT',
       'BUILDING_CLASS_AT_PRESENT', 'ADDRESS', 'APARTMENT_NUMBER', 'ZIP_CODE',
       'RESIDENTIAL_UNITS', 'COMMERCIAL_UNITS', 'TOTAL_UNITS',
       'LAND_SQUARE_FEET', 'GROSS_SQUARE_FEET', 'YEAR_BUILT',
       'TAX_CLASS_AT_TIME_OF_SALE', 'BUILDING_CLASS_AT_TIME_OF_SALE',
       'SALE_PRICE', 'SALE_DATE']

#Pull Sq ft data from Zillow Median Square Footage Excel File 
zillow_squarefootage=pd.read_excel("Zip_MedianListingPricePerSqft_AllHomes.xls")
zillow_squarefootage=zillow_squarefootage.loc[:,["RegionName","2018-10"]] 
zillow_squarefootage['ZIP_CODE']=zillow_squarefootage['RegionName']
zillow_squarefootage['ZillowSquareFootage']=zillow_squarefootage['2018-10']

#Pull Median Price data from Zillow Median Price Excel File
zillow_median_listing=pd.read_excel("Zip_MedianListingPrice_AllHomes.xls")
zillow_median_listing=zillow_median_listing.loc[:,["RegionName","2018-10"]] 
zillow_median_listing['ZIP_CODE']=zillow_median_listing['RegionName']
zillow_median_listing['ZillowMedianPrice']=zillow_median_listing['2018-10']

#Merge Zillow data together
new_df2= zillow_squarefootage.merge(zillow_median_listing, how = 'inner', on = ['ZIP_CODE'])
# new_df3=new_df2.merge(average_by_zip_2018, how = 'inner', on = ['ZIP_CODE'])

#Merge merged Zillow data with main excel Rolling Sales Data
new_df3= data.merge(new_df2, how = 'inner', on = ['ZIP_CODE'])
new_df=new_df3.copy()
new_df=new_df.drop(columns=['RegionName_y', 'RegionName_x',"2018-10_x","2018-10_y"])
new_df.columns
```




    Index(['BOROUGH', 'NEIGHBORHOOD', 'BUILDING_CLASS_CATEGORY',
           'TAX_CLASS_AT_PRESENT', 'BLOCK', 'LOT', 'EASE-MENT',
           'BUILDING_CLASS_AT_PRESENT', 'ADDRESS', 'APARTMENT_NUMBER', 'ZIP_CODE',
           'RESIDENTIAL_UNITS', 'COMMERCIAL_UNITS', 'TOTAL_UNITS',
           'LAND_SQUARE_FEET', 'GROSS_SQUARE_FEET', 'YEAR_BUILT',
           'TAX_CLASS_AT_TIME_OF_SALE', 'BUILDING_CLASS_AT_TIME_OF_SALE',
           'SALE_PRICE', 'SALE_DATE', 'ZillowSquareFootage', 'ZillowMedianPrice'],
          dtype='object')




```python
new_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BOROUGH</th>
      <th>NEIGHBORHOOD</th>
      <th>BUILDING_CLASS_CATEGORY</th>
      <th>TAX_CLASS_AT_PRESENT</th>
      <th>BLOCK</th>
      <th>LOT</th>
      <th>EASE-MENT</th>
      <th>BUILDING_CLASS_AT_PRESENT</th>
      <th>ADDRESS</th>
      <th>APARTMENT_NUMBER</th>
      <th>...</th>
      <th>TOTAL_UNITS</th>
      <th>LAND_SQUARE_FEET</th>
      <th>GROSS_SQUARE_FEET</th>
      <th>YEAR_BUILT</th>
      <th>TAX_CLASS_AT_TIME_OF_SALE</th>
      <th>BUILDING_CLASS_AT_TIME_OF_SALE</th>
      <th>SALE_PRICE</th>
      <th>SALE_DATE</th>
      <th>ZillowSquareFootage</th>
      <th>ZillowMedianPrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>CLINTON</td>
      <td>08 RENTALS - ELEVATOR APARTMENTS</td>
      <td>2</td>
      <td>1071</td>
      <td>42</td>
      <td></td>
      <td>D6</td>
      <td>520 WEST 43RD STREET</td>
      <td></td>
      <td>...</td>
      <td>380</td>
      <td>24100</td>
      <td>319967</td>
      <td>1996</td>
      <td>2</td>
      <td>D6</td>
      <td>193000000</td>
      <td>2018-08-01</td>
      <td>1659.125189</td>
      <td>1310000.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>CLINTON</td>
      <td>29 COMMERCIAL GARAGES</td>
      <td>4</td>
      <td>1076</td>
      <td>1</td>
      <td></td>
      <td>G8</td>
      <td>646 11TH AVENUE</td>
      <td></td>
      <td>...</td>
      <td>2</td>
      <td>30125</td>
      <td>50207</td>
      <td>1946</td>
      <td>4</td>
      <td>G8</td>
      <td>93125000</td>
      <td>2018-06-08</td>
      <td>1659.125189</td>
      <td>1310000.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>MIDTOWN WEST</td>
      <td>21 OFFICE BUILDINGS</td>
      <td>4</td>
      <td>1263</td>
      <td>56</td>
      <td></td>
      <td>O5</td>
      <td>32 WEST 48TH STREET</td>
      <td></td>
      <td>...</td>
      <td>61</td>
      <td>8234</td>
      <td>33837</td>
      <td>1924</td>
      <td>4</td>
      <td>O5</td>
      <td>40000000</td>
      <td>2018-10-26</td>
      <td>1659.125189</td>
      <td>1310000.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>CLINTON</td>
      <td>26 OTHER HOTELS</td>
      <td>4</td>
      <td>1076</td>
      <td>57</td>
      <td></td>
      <td>H3</td>
      <td>548 WEST 48TH   STREET</td>
      <td></td>
      <td>...</td>
      <td>1</td>
      <td>5021</td>
      <td>28540</td>
      <td>2010</td>
      <td>4</td>
      <td>H3</td>
      <td>29168078</td>
      <td>2017-12-27</td>
      <td>1659.125189</td>
      <td>1310000.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>CLINTON</td>
      <td>07 RENTALS - WALKUP APARTMENTS</td>
      <td>2</td>
      <td>1055</td>
      <td>55</td>
      <td></td>
      <td>C4</td>
      <td>444 WEST 46TH STREET</td>
      <td></td>
      <td>...</td>
      <td>20</td>
      <td>2510</td>
      <td>9810</td>
      <td>1901</td>
      <td>2</td>
      <td>C4</td>
      <td>2643864</td>
      <td>2018-09-21</td>
      <td>1659.125189</td>
      <td>1310000.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>



## Data Cleaning and Deciding what features to use


```python
#Change the datatype of Zillow Median Price and Square Footage from Float to Integer
new_df['ZillowMedianPrice'] = new_df['ZillowMedianPrice'].astype(int)
new_df['ZillowSquareFootage'] = new_df['ZillowSquareFootage'].astype(int)

#Set target (Y value) = sales price / #Set features (X values) = all columns (will drop all unecessary)
target=new_df[["SALE_PRICE"]]
features= new_df 

#Drop unecessary features, maybe drop both Zillow data columns
features=features.drop(columns=["SALE_DATE",'BOROUGH',                     
       'TAX_CLASS_AT_PRESENT', 'BLOCK', 'LOT', 'EASE-MENT',
       'BUILDING_CLASS_AT_PRESENT', 'ADDRESS', 'APARTMENT_NUMBER', 
        'TOTAL_UNITS',
       'TAX_CLASS_AT_TIME_OF_SALE', 'BUILDING_CLASS_AT_TIME_OF_SALE',
      'SALE_PRICE','ZillowSquareFootage', 'ZillowMedianPrice',]) 

##Data Cleaning
#Strip duplicate BUILDING CLASS CATEGORY and NEIGHBORHOOD categories
features["BUILDING_CLASS_CATEGORY"]=features["BUILDING_CLASS_CATEGORY"].str.strip()
features["BUILDING_CLASS_CATEGORY"]=features["BUILDING_CLASS_CATEGORY"].str.replace(' ', '')
features["NEIGHBORHOOD"]=features["NEIGHBORHOOD"].str.strip()
features["NEIGHBORHOOD"]=features["NEIGHBORHOOD"].str.replace(' ', '')

#Set Category Variables
cat_vars=features[['BUILDING_CLASS_CATEGORY',"NEIGHBORHOOD","ZIP_CODE"]]
```


```python
features.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NEIGHBORHOOD</th>
      <th>BUILDING_CLASS_CATEGORY</th>
      <th>ZIP_CODE</th>
      <th>RESIDENTIAL_UNITS</th>
      <th>COMMERCIAL_UNITS</th>
      <th>LAND_SQUARE_FEET</th>
      <th>GROSS_SQUARE_FEET</th>
      <th>YEAR_BUILT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CLINTON</td>
      <td>08RENTALS-ELEVATORAPARTMENTS</td>
      <td>10036</td>
      <td>375</td>
      <td>5</td>
      <td>24100</td>
      <td>319967</td>
      <td>1996</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CLINTON</td>
      <td>29COMMERCIALGARAGES</td>
      <td>10036</td>
      <td>0</td>
      <td>2</td>
      <td>30125</td>
      <td>50207</td>
      <td>1946</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MIDTOWNWEST</td>
      <td>21OFFICEBUILDINGS</td>
      <td>10036</td>
      <td>0</td>
      <td>61</td>
      <td>8234</td>
      <td>33837</td>
      <td>1924</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CLINTON</td>
      <td>26OTHERHOTELS</td>
      <td>10036</td>
      <td>0</td>
      <td>1</td>
      <td>5021</td>
      <td>28540</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CLINTON</td>
      <td>07RENTALS-WALKUPAPARTMENTS</td>
      <td>10036</td>
      <td>20</td>
      <td>0</td>
      <td>2510</td>
      <td>9810</td>
      <td>1901</td>
    </tr>
  </tbody>
</table>
</div>



## Change Category Variables to Dummy Variables


```python
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(features[var], prefix=var)#,drop_first=True)
    data1=features.join(cat_list)
    features=data1
data_vars=features.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]
features=features[to_keep]

```



## Graph out Sale Price Distribution


```python
new_df["SALE_PRICE"].describe()
sns.distplot(new_df.SALE_PRICE,fit=norm);
plt.ylabel =('Frequency')
plt.title = ('SalePrice Distribution');
#Get the fitted parameters used by the function
(mu, sigma) = norm.fit(new_df["SALE_PRICE"]);
#QQ plot
fig = plt.figure()
res = stats.probplot(new_df["SALE_PRICE"], plot=plt)
# plt.show()
print("skewness: %f" % new_df["SALE_PRICE"].skew())
print("kurtosis: %f" % new_df["SALE_PRICE"].kurt())
```

    /Users/chrischung/anaconda3/lib/python3.7/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval


    skewness: 4.750264
    kurtosis: 26.823504



![png](PredictAPrice%20-%20Notebook_files/PredictAPrice%20-%20Notebook_13_2.png)



![png](PredictAPrice%20-%20Notebook_files/PredictAPrice%20-%20Notebook_13_3.png)


## Transform Data


```python
#notes
#Plotted the distribution of the SALE_PRICE and normal probability graph which is used to identify substantive departures from normality. This includes identifying outliers, skewness and kurtosis. Used the QQ-plot
#log transform the target 
new_df["SALE_PRICE"] = np.log1p(new_df["SALE_PRICE"])

#Kernel Density plot
sns.distplot(new_df.SALE_PRICE,fit=norm);
plt.ylabel=('Frequency')
plt.title=('SalePrice distribution');
#Get the fitted parameters used by the function
(mu,sigma)= norm.fit(new_df["SALE_PRICE"]);
#QQ plot
fig =plt.figure()
res =stats. probplot(new_df["SALE_PRICE"], plot=plt)
plt.show()

```

    /Users/chrischung/anaconda3/lib/python3.7/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval



![png](PredictAPrice%20-%20Notebook_files/PredictAPrice%20-%20Notebook_15_1.png)



![png](PredictAPrice%20-%20Notebook_files/PredictAPrice%20-%20Notebook_15_2.png)


# # Step 1: Checking for Linearity  using Scatterplots


```python

sns.regplot(y=new_df.SALE_PRICE, x=new_df['RESIDENTIAL_UNITS'], data=new_df, fit_reg = True)

```

    /Users/chrischung/anaconda3/lib/python3.7/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval





    <matplotlib.axes._subplots.AxesSubplot at 0x1c2953d438>




![png](PredictAPrice%20-%20Notebook_files/PredictAPrice%20-%20Notebook_17_2.png)



```python
sns.regplot(y=new_df.SALE_PRICE, x=new_df['COMMERCIAL_UNITS'], data=new_df, fit_reg = True)


```

    /Users/chrischung/anaconda3/lib/python3.7/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval





    <matplotlib.axes._subplots.AxesSubplot at 0x1c295253c8>




![png](PredictAPrice%20-%20Notebook_files/PredictAPrice%20-%20Notebook_18_2.png)



```python
sns.regplot(y=new_df.SALE_PRICE, x=new_df['LAND_SQUARE_FEET'], data=new_df, fit_reg = True)
```

    /Users/chrischung/anaconda3/lib/python3.7/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval





    <matplotlib.axes._subplots.AxesSubplot at 0x1c2964a518>




![png](PredictAPrice%20-%20Notebook_files/PredictAPrice%20-%20Notebook_19_2.png)



```python
sns.regplot(y=new_df.SALE_PRICE, x=new_df['GROSS_SQUARE_FEET'], data=new_df, fit_reg = True)

```

    /Users/chrischung/anaconda3/lib/python3.7/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval





    <matplotlib.axes._subplots.AxesSubplot at 0x1c2962e5f8>




![png](PredictAPrice%20-%20Notebook_files/PredictAPrice%20-%20Notebook_20_2.png)


# Graph showing the distribution of prices by neighborhoood


```python

plt.figure(figsize=(20,7))
sns.stripplot(x = new_df.NEIGHBORHOOD, y = new_df.SALE_PRICE,
              order = np.sort(new_df.NEIGHBORHOOD.unique()),
              jitter=0.1, alpha=0.5)
plt.xticks(rotation=90)



```




    (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
            34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]),
     <a list of 45 Text xticklabel objects>)




![png](PredictAPrice%20-%20Notebook_files/PredictAPrice%20-%20Notebook_22_1.png)


# Distribution of square footage and sale price


```python

plt.figure(figsize=(12,7))
sns.stripplot(x = new_df.GROSS_SQUARE_FEET, y = new_df.SALE_PRICE,
              order = np.sort(new_df.GROSS_SQUARE_FEET),
              jitter=0.1, alpha=0.5)
plt.xticks(rotation=45)
```




    (array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
             13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
             26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
             39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
             52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,
             65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
             78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,
             91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103,
            104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
            117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
            130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
            143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
            156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168,
            169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181,
            182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194,
            195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
            208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220,
            221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233,
            234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246,
            247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259,
            260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272,
            273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285,
            286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298,
            299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311,
            312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324,
            325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337,
            338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350,
            351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363,
            364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376,
            377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389,
            390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402,
            403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415,
            416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428,
            429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441,
            442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454,
            455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467,
            468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480,
            481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493,
            494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506,
            507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519,
            520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532,
            533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545,
            546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558,
            559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571,
            572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584,
            585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597,
            598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610,
            611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623,
            624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636,
            637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649,
            650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662,
            663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675,
            676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688,
            689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701,
            702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714,
            715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727,
            728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740,
            741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753,
            754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766,
            767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779,
            780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792,
            793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805,
            806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818,
            819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831,
            832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844,
            845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857,
            858, 859, 860, 861, 862, 863, 864, 865]),
     <a list of 866 Text xticklabel objects>)




![png](PredictAPrice%20-%20Notebook_files/PredictAPrice%20-%20Notebook_24_1.png)


# Distribution of homes built by year 


```python
var  = 'YEAR_BUILT'
data= pd.concat([new_df['SALE_PRICE'], new_df[var]], axis =1)
f, ax = plt.subplots(figsize=(16, 16))
fig = sns.boxplot(x=var, y=new_df['SALE_PRICE'], data=data)
fig.axis(ymin=5)
plt.xticks(rotation=90);
plt.show();
```


![png](PredictAPrice%20-%20Notebook_files/PredictAPrice%20-%20Notebook_26_0.png)


## Correlation of Features


```python
features.corr() >.75
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ZIP_CODE</th>
      <th>RESIDENTIAL_UNITS</th>
      <th>COMMERCIAL_UNITS</th>
      <th>LAND_SQUARE_FEET</th>
      <th>GROSS_SQUARE_FEET</th>
      <th>YEAR_BUILT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ZIP_CODE</th>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>RESIDENTIAL_UNITS</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>COMMERCIAL_UNITS</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>LAND_SQUARE_FEET</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>GROSS_SQUARE_FEET</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>YEAR_BUILT</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
import seaborn as sns
sns.heatmap(features.corr(), center=0);
```


![png](PredictAPrice%20-%20Notebook_files/PredictAPrice%20-%20Notebook_29_0.png)


# Regression Info Below
# Features described first 


```python

features.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RESIDENTIAL_UNITS</th>
      <th>COMMERCIAL_UNITS</th>
      <th>LAND_SQUARE_FEET</th>
      <th>GROSS_SQUARE_FEET</th>
      <th>YEAR_BUILT</th>
      <th>BUILDING_CLASS_CATEGORY_01ONEFAMILYDWELLINGS</th>
      <th>BUILDING_CLASS_CATEGORY_02TWOFAMILYDWELLINGS</th>
      <th>BUILDING_CLASS_CATEGORY_03THREEFAMILYDWELLINGS</th>
      <th>BUILDING_CLASS_CATEGORY_07RENTALS-WALKUPAPARTMENTS</th>
      <th>BUILDING_CLASS_CATEGORY_08RENTALS-ELEVATORAPARTMENTS</th>
      <th>...</th>
      <th>ZIP_CODE_10009</th>
      <th>ZIP_CODE_10012</th>
      <th>ZIP_CODE_10013</th>
      <th>ZIP_CODE_10016</th>
      <th>ZIP_CODE_10018</th>
      <th>ZIP_CODE_10019</th>
      <th>ZIP_CODE_10027</th>
      <th>ZIP_CODE_10029</th>
      <th>ZIP_CODE_10036</th>
      <th>ZIP_CODE_10038</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>866.000000</td>
      <td>866.000000</td>
      <td>866.000000</td>
      <td>8.660000e+02</td>
      <td>866.000000</td>
      <td>866.000000</td>
      <td>866.000000</td>
      <td>866.000000</td>
      <td>866.000000</td>
      <td>866.000000</td>
      <td>...</td>
      <td>866.000000</td>
      <td>866.000000</td>
      <td>866.000000</td>
      <td>866.000000</td>
      <td>866.000000</td>
      <td>866.000000</td>
      <td>866.000000</td>
      <td>866.000000</td>
      <td>866.000000</td>
      <td>866.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>9.247113</td>
      <td>2.975751</td>
      <td>5549.627021</td>
      <td>5.932070e+04</td>
      <td>1952.683603</td>
      <td>0.032333</td>
      <td>0.026559</td>
      <td>0.021940</td>
      <td>0.259815</td>
      <td>0.049654</td>
      <td>...</td>
      <td>0.060046</td>
      <td>0.028868</td>
      <td>0.080831</td>
      <td>0.066975</td>
      <td>0.033487</td>
      <td>0.420323</td>
      <td>0.112009</td>
      <td>0.100462</td>
      <td>0.025404</td>
      <td>0.004619</td>
    </tr>
    <tr>
      <th>std</th>
      <td>31.859502</td>
      <td>12.372564</td>
      <td>6301.376164</td>
      <td>9.459625e+04</td>
      <td>48.081714</td>
      <td>0.176984</td>
      <td>0.160883</td>
      <td>0.146572</td>
      <td>0.438787</td>
      <td>0.217354</td>
      <td>...</td>
      <td>0.237710</td>
      <td>0.167533</td>
      <td>0.272734</td>
      <td>0.250122</td>
      <td>0.180009</td>
      <td>0.493896</td>
      <td>0.315560</td>
      <td>0.300789</td>
      <td>0.157440</td>
      <td>0.067845</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.360000e+02</td>
      <td>1800.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2124.250000</td>
      <td>6.251500e+03</td>
      <td>1910.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>4966.500000</td>
      <td>2.035950e+04</td>
      <td>1925.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>8.000000</td>
      <td>2.000000</td>
      <td>7532.000000</td>
      <td>1.128500e+05</td>
      <td>2007.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>476.000000</td>
      <td>292.000000</td>
      <td>80333.000000</td>
      <td>1.613847e+06</td>
      <td>2016.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 65 columns</p>
</div>



# Of the 5 original features used [RESIDENTIAL_UNITS, COMMERCIAL_UNITS, LAND_SQUARE_FEET, GROSS_SQUARE_FEET and YEAR_BUILT], only the LAND_SQUARE_FEET had a P value above 0.05. Its value was 0.219

# R^2 total using 4 features = 0.727 


```python
# GROSS_SQUARE_FEET:0.645
# Residential Units: 0.071
# COMMERCIAL_UNITS :0.098
# LAND_SQUARE_FEET:0.534
# YEAR_BUILT:0.000
#ZillowSquareFootage:0.006... pvalue of 0.024
#ZillowMedianPrice0.012...pvalue of 0.001
#ALL WITH ZILLOW: 0.729

m7 = ols('SALE_PRICE ~RESIDENTIAL_UNITS+COMMERCIAL_UNITS+GROSS_SQUARE_FEET+YEAR_BUILT',new_df).fit()
print(m7.summary())
m1 = ols('SALE_PRICE ~GROSS_SQUARE_FEET',new_df).fit()
print(m1.summary())
# m2 = ols('SALE_PRICE ~RESIDENTIAL_UNITS ',new_df).fit()
# print(m2.summary())
# m3 = ols('SALE_PRICE ~COMMERCIAL_UNITS ',new_df).fit()
# print(m3.summary())
# m4 = ols('SALE_PRICE ~LAND_SQUARE_FEET ',new_df).fit()
# print(m4.summary())
m5 = ols('SALE_PRICE ~GROSS_SQUARE_FEET ',new_df).fit()
print(m1.summary())
# m6 = ols('SALE_PRICE ~YEAR_BUILT ',new_df).fit()
# print(m6.summary())


#####regression for zillow items below 
# m8 = ols('SALE_PRICE ~ZillowSquareFootage ',new_df).fit()
# print(m8.summary())
# m9 = ols('SALE_PRICE ~ZillowMedianPrice ',new_df).fit()
# print(m9.summary())
# m10 = ols('SALE_PRICE ~RESIDENTIAL_UNITS+ZillowSquareFootage+ZillowMedianPrice+COMMERCIAL_UNITS+LAND_SQUARE_FEET+GROSS_SQUARE_FEET+YEAR_BUILT ',new_df).fit()
# print(m10.summary())

1- (RSS/TSS)
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:             SALE_PRICE   R-squared:                       0.602
    Model:                            OLS   Adj. R-squared:                  0.601
    Method:                 Least Squares   F-statistic:                     326.2
    Date:                Fri, 07 Dec 2018   Prob (F-statistic):          8.86e-171
    Time:                        13:33:48   Log-Likelihood:                -1653.8
    No. Observations:                 866   AIC:                             3318.
    Df Residuals:                     861   BIC:                             3341.
    Df Model:                           4                                         
    Covariance Type:            nonrobust                                         
    =====================================================================================
                            coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------------
    Intercept            84.9087      2.607     32.565      0.000      79.791      90.026
    RESIDENTIAL_UNITS     0.0208      0.002     11.541      0.000       0.017       0.024
    COMMERCIAL_UNITS      0.0424      0.005      8.798      0.000       0.033       0.052
    GROSS_SQUARE_FEET -1.034e-06   7.26e-07     -1.423      0.155   -2.46e-06    3.92e-07
    YEAR_BUILT           -0.0364      0.001    -27.134      0.000      -0.039      -0.034
    ==============================================================================
    Omnibus:                      138.842   Durbin-Watson:                   1.210
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1003.100
    Skew:                           0.503   Prob(JB):                    1.51e-218
    Kurtosis:                       8.176   Cond. No.                     5.23e+06
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 5.23e+06. This might indicate that there are
    strong multicollinearity or other numerical problems.
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:             SALE_PRICE   R-squared:                       0.061
    Model:                            OLS   Adj. R-squared:                  0.060
    Method:                 Least Squares   F-statistic:                     56.30
    Date:                Fri, 07 Dec 2018   Prob (F-statistic):           1.55e-13
    Time:                        13:33:48   Log-Likelihood:                -2025.9
    No. Observations:                 866   AIC:                             4056.
    Df Residuals:                     864   BIC:                             4065.
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    =====================================================================================
                            coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------------
    Intercept            14.4053      0.101    142.878      0.000      14.207      14.603
    GROSS_SQUARE_FEET -6.778e-06   9.03e-07     -7.503      0.000   -8.55e-06      -5e-06
    ==============================================================================
    Omnibus:                       99.204   Durbin-Watson:                   0.360
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              276.513
    Skew:                           0.587   Prob(JB):                     9.03e-61
    Kurtosis:                       5.507   Cond. No.                     1.32e+05
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.32e+05. This might indicate that there are
    strong multicollinearity or other numerical problems.
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:             SALE_PRICE   R-squared:                       0.061
    Model:                            OLS   Adj. R-squared:                  0.060
    Method:                 Least Squares   F-statistic:                     56.30
    Date:                Fri, 07 Dec 2018   Prob (F-statistic):           1.55e-13
    Time:                        13:33:48   Log-Likelihood:                -2025.9
    No. Observations:                 866   AIC:                             4056.
    Df Residuals:                     864   BIC:                             4065.
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    =====================================================================================
                            coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------------
    Intercept            14.4053      0.101    142.878      0.000      14.207      14.603
    GROSS_SQUARE_FEET -6.778e-06   9.03e-07     -7.503      0.000   -8.55e-06      -5e-06
    ==============================================================================
    Omnibus:                       99.204   Durbin-Watson:                   0.360
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              276.513
    Skew:                           0.587   Prob(JB):                     9.03e-61
    Kurtosis:                       5.507   Cond. No.                     1.32e+05
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.32e+05. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
reg = LinearRegression()
```


```python
new_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BOROUGH</th>
      <th>NEIGHBORHOOD</th>
      <th>BUILDING_CLASS_CATEGORY</th>
      <th>TAX_CLASS_AT_PRESENT</th>
      <th>BLOCK</th>
      <th>LOT</th>
      <th>EASE-MENT</th>
      <th>BUILDING_CLASS_AT_PRESENT</th>
      <th>ADDRESS</th>
      <th>APARTMENT_NUMBER</th>
      <th>...</th>
      <th>TOTAL_UNITS</th>
      <th>LAND_SQUARE_FEET</th>
      <th>GROSS_SQUARE_FEET</th>
      <th>YEAR_BUILT</th>
      <th>TAX_CLASS_AT_TIME_OF_SALE</th>
      <th>BUILDING_CLASS_AT_TIME_OF_SALE</th>
      <th>SALE_PRICE</th>
      <th>SALE_DATE</th>
      <th>ZillowSquareFootage</th>
      <th>ZillowMedianPrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>CLINTON</td>
      <td>08 RENTALS - ELEVATOR APARTMENTS</td>
      <td>2</td>
      <td>1071</td>
      <td>42</td>
      <td></td>
      <td>D6</td>
      <td>520 WEST 43RD STREET</td>
      <td></td>
      <td>...</td>
      <td>380</td>
      <td>24100</td>
      <td>319967</td>
      <td>1996</td>
      <td>2</td>
      <td>D6</td>
      <td>19.078201</td>
      <td>2018-08-01</td>
      <td>1659</td>
      <td>1310000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>CLINTON</td>
      <td>29 COMMERCIAL GARAGES</td>
      <td>4</td>
      <td>1076</td>
      <td>1</td>
      <td></td>
      <td>G8</td>
      <td>646 11TH AVENUE</td>
      <td></td>
      <td>...</td>
      <td>2</td>
      <td>30125</td>
      <td>50207</td>
      <td>1946</td>
      <td>4</td>
      <td>G8</td>
      <td>18.349453</td>
      <td>2018-06-08</td>
      <td>1659</td>
      <td>1310000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>MIDTOWN WEST</td>
      <td>21 OFFICE BUILDINGS</td>
      <td>4</td>
      <td>1263</td>
      <td>56</td>
      <td></td>
      <td>O5</td>
      <td>32 WEST 48TH STREET</td>
      <td></td>
      <td>...</td>
      <td>61</td>
      <td>8234</td>
      <td>33837</td>
      <td>1924</td>
      <td>4</td>
      <td>O5</td>
      <td>17.504390</td>
      <td>2018-10-26</td>
      <td>1659</td>
      <td>1310000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>CLINTON</td>
      <td>26 OTHER HOTELS</td>
      <td>4</td>
      <td>1076</td>
      <td>57</td>
      <td></td>
      <td>H3</td>
      <td>548 WEST 48TH   STREET</td>
      <td></td>
      <td>...</td>
      <td>1</td>
      <td>5021</td>
      <td>28540</td>
      <td>2010</td>
      <td>4</td>
      <td>H3</td>
      <td>17.188585</td>
      <td>2017-12-27</td>
      <td>1659</td>
      <td>1310000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>CLINTON</td>
      <td>07 RENTALS - WALKUP APARTMENTS</td>
      <td>2</td>
      <td>1055</td>
      <td>55</td>
      <td></td>
      <td>C4</td>
      <td>444 WEST 46TH STREET</td>
      <td></td>
      <td>...</td>
      <td>20</td>
      <td>2510</td>
      <td>9810</td>
      <td>1901</td>
      <td>2</td>
      <td>C4</td>
      <td>14.787752</td>
      <td>2018-09-21</td>
      <td>1659</td>
      <td>1310000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>MIDTOWN WEST</td>
      <td>22 STORE BUILDINGS</td>
      <td>4</td>
      <td>1034</td>
      <td>31</td>
      <td></td>
      <td>K4</td>
      <td>687 8 AVENUE</td>
      <td></td>
      <td>...</td>
      <td>5</td>
      <td>2017</td>
      <td>7760</td>
      <td>1920</td>
      <td>4</td>
      <td>K4</td>
      <td>16.436549</td>
      <td>2018-05-16</td>
      <td>1659</td>
      <td>1310000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>CLINTON</td>
      <td>07 RENTALS - WALKUP APARTMENTS</td>
      <td>2</td>
      <td>1053</td>
      <td>6</td>
      <td></td>
      <td>C4</td>
      <td>459 WEST 43RD   STREET</td>
      <td></td>
      <td>...</td>
      <td>13</td>
      <td>2510</td>
      <td>6640</td>
      <td>1910</td>
      <td>2</td>
      <td>C4</td>
      <td>16.231424</td>
      <td>2018-01-24</td>
      <td>1659</td>
      <td>1310000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>CLINTON</td>
      <td>07 RENTALS - WALKUP APARTMENTS</td>
      <td>2A</td>
      <td>1058</td>
      <td>113</td>
      <td></td>
      <td>C3</td>
      <td>435 WEST 48TH STREET</td>
      <td></td>
      <td>...</td>
      <td>4</td>
      <td>2510</td>
      <td>6560</td>
      <td>1910</td>
      <td>2</td>
      <td>C3</td>
      <td>15.706361</td>
      <td>2018-07-30</td>
      <td>1659</td>
      <td>1310000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>MIDTOWN WEST</td>
      <td>22 STORE BUILDINGS</td>
      <td>4</td>
      <td>1263</td>
      <td>55</td>
      <td></td>
      <td>K9</td>
      <td>30 WEST 48TH STREET</td>
      <td></td>
      <td>...</td>
      <td>5</td>
      <td>2070</td>
      <td>6350</td>
      <td>1920</td>
      <td>4</td>
      <td>K9</td>
      <td>16.146975</td>
      <td>2018-10-26</td>
      <td>1659</td>
      <td>1310000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>MIDTOWN WEST</td>
      <td>22 STORE BUILDINGS</td>
      <td>4</td>
      <td>1263</td>
      <td>21</td>
      <td></td>
      <td>K2</td>
      <td>25 WEST 47TH STREET</td>
      <td></td>
      <td>...</td>
      <td>26</td>
      <td>2397</td>
      <td>5643</td>
      <td>1938</td>
      <td>4</td>
      <td>K2</td>
      <td>16.299348</td>
      <td>2018-10-26</td>
      <td>1659</td>
      <td>1310000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>MIDTOWN WEST</td>
      <td>22 STORE BUILDINGS</td>
      <td>4</td>
      <td>999</td>
      <td>11</td>
      <td></td>
      <td>K4</td>
      <td>149 WEST 46TH STREET</td>
      <td></td>
      <td>...</td>
      <td>3</td>
      <td>1883</td>
      <td>3982</td>
      <td>1920</td>
      <td>4</td>
      <td>K4</td>
      <td>16.929026</td>
      <td>2018-10-23</td>
      <td>1659</td>
      <td>1310000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>CLINTON</td>
      <td>02 TWO FAMILY DWELLINGS</td>
      <td>1</td>
      <td>1053</td>
      <td>55</td>
      <td></td>
      <td>S2</td>
      <td>448 WEST 44TH STREET</td>
      <td></td>
      <td>...</td>
      <td>3</td>
      <td>1883</td>
      <td>3600</td>
      <td>1899</td>
      <td>1</td>
      <td>S2</td>
      <td>14.800354</td>
      <td>2018-01-19</td>
      <td>1659</td>
      <td>1310000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>MIDTOWN WEST</td>
      <td>38 ASYLUMS AND HOMES</td>
      <td>4</td>
      <td>1039</td>
      <td>123</td>
      <td></td>
      <td>N9</td>
      <td>317 WEST 48TH   STREET</td>
      <td></td>
      <td>...</td>
      <td>0</td>
      <td>1808</td>
      <td>3200</td>
      <td>1920</td>
      <td>4</td>
      <td>N9</td>
      <td>15.150512</td>
      <td>2018-03-23</td>
      <td>1659</td>
      <td>1310000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>CLINTON</td>
      <td>29  COMMERCIAL GARAGES</td>
      <td>4</td>
      <td>1095</td>
      <td>24</td>
      <td></td>
      <td>G2</td>
      <td>607-611 WEST 47TH   STREET</td>
      <td></td>
      <td>...</td>
      <td>2</td>
      <td>7531</td>
      <td>15062</td>
      <td>1940</td>
      <td>4</td>
      <td>G2</td>
      <td>16.733281</td>
      <td>2017-01-20</td>
      <td>1659</td>
      <td>1310000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1</td>
      <td>CLINTON</td>
      <td>37  RELIGIOUS FACILITIES</td>
      <td>4</td>
      <td>1053</td>
      <td>59</td>
      <td></td>
      <td>M4</td>
      <td>460 WEST 44TH   STREET</td>
      <td></td>
      <td>...</td>
      <td>2</td>
      <td>5021</td>
      <td>8848</td>
      <td>1910</td>
      <td>4</td>
      <td>M4</td>
      <td>15.896201</td>
      <td>2017-03-17</td>
      <td>1659</td>
      <td>1310000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>MIDTOWN WEST</td>
      <td>07  RENTALS - WALKUP APARTMENTS</td>
      <td>2</td>
      <td>1036</td>
      <td>45</td>
      <td></td>
      <td>C7</td>
      <td>328 WEST 46TH   STREET</td>
      <td></td>
      <td>...</td>
      <td>14</td>
      <td>4017</td>
      <td>7316</td>
      <td>1920</td>
      <td>2</td>
      <td>C7</td>
      <td>16.341239</td>
      <td>2017-01-27</td>
      <td>1659</td>
      <td>1310000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1</td>
      <td>MIDTOWN WEST</td>
      <td>21  OFFICE BUILDINGS</td>
      <td>4</td>
      <td>1260</td>
      <td>1</td>
      <td></td>
      <td>O4</td>
      <td>1140 AVENUE OF THE AMER</td>
      <td></td>
      <td>...</td>
      <td>84</td>
      <td>9375</td>
      <td>179513</td>
      <td>1931</td>
      <td>4</td>
      <td>O4</td>
      <td>19.008467</td>
      <td>2016-06-15</td>
      <td>1659</td>
      <td>1310000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>MIDTOWN WEST</td>
      <td>21  OFFICE BUILDINGS</td>
      <td>4</td>
      <td>1260</td>
      <td>64</td>
      <td></td>
      <td>O6</td>
      <td>56 WEST 45TH   STREET</td>
      <td></td>
      <td>...</td>
      <td>62</td>
      <td>4016</td>
      <td>58538</td>
      <td>1914</td>
      <td>4</td>
      <td>O6</td>
      <td>17.727534</td>
      <td>2017-01-05</td>
      <td>1659</td>
      <td>1310000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1</td>
      <td>MIDTOWN WEST</td>
      <td>21  OFFICE BUILDINGS</td>
      <td>4</td>
      <td>1263</td>
      <td>1</td>
      <td></td>
      <td>O5</td>
      <td>1200 AVENUE OF THE AMER</td>
      <td></td>
      <td>...</td>
      <td>42</td>
      <td>2900</td>
      <td>21600</td>
      <td>1960</td>
      <td>4</td>
      <td>O5</td>
      <td>17.579034</td>
      <td>2017-04-14</td>
      <td>1659</td>
      <td>1310000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>MIDTOWN WEST</td>
      <td>22  STORE BUILDINGS</td>
      <td>4</td>
      <td>1001</td>
      <td>11</td>
      <td></td>
      <td>K9</td>
      <td>151 WEST 48TH   STREET</td>
      <td></td>
      <td>...</td>
      <td>1</td>
      <td>1903</td>
      <td>10167</td>
      <td>1920</td>
      <td>4</td>
      <td>K9</td>
      <td>16.381267</td>
      <td>2016-09-01</td>
      <td>1659</td>
      <td>1310000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1</td>
      <td>MIDTOWN WEST</td>
      <td>25  LUXURY HOTELS</td>
      <td>4</td>
      <td>1260</td>
      <td>56</td>
      <td></td>
      <td>H2</td>
      <td>40 WEST 45TH   STREET</td>
      <td></td>
      <td>...</td>
      <td>1</td>
      <td>6025</td>
      <td>58935</td>
      <td>1903</td>
      <td>4</td>
      <td>H2</td>
      <td>17.887950</td>
      <td>2016-08-25</td>
      <td>1659</td>
      <td>1310000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1</td>
      <td>MIDTOWN WEST</td>
      <td>29  COMMERCIAL GARAGES</td>
      <td>4</td>
      <td>1263</td>
      <td>45</td>
      <td></td>
      <td>G1</td>
      <td>10 WEST 48TH   STREET</td>
      <td></td>
      <td>...</td>
      <td>2</td>
      <td>7531</td>
      <td>51403</td>
      <td>1966</td>
      <td>4</td>
      <td>G1</td>
      <td>17.431819</td>
      <td>2016-06-09</td>
      <td>1659</td>
      <td>1310000</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1</td>
      <td>LOWER EAST SIDE</td>
      <td>08 RENTALS - ELEVATOR APARTMENTS</td>
      <td>2</td>
      <td>246</td>
      <td>1</td>
      <td></td>
      <td>D6</td>
      <td>275 SOUTH STREET</td>
      <td></td>
      <td>...</td>
      <td>258</td>
      <td>51180</td>
      <td>262875</td>
      <td>1978</td>
      <td>2</td>
      <td>D6</td>
      <td>18.936494</td>
      <td>2017-12-14</td>
      <td>2030</td>
      <td>1550500</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1</td>
      <td>LOWER EAST SIDE</td>
      <td>08 RENTALS - ELEVATOR APARTMENTS</td>
      <td>2</td>
      <td>283</td>
      <td>24</td>
      <td></td>
      <td>D7</td>
      <td>10 RUTGERS STREET</td>
      <td></td>
      <td>...</td>
      <td>89</td>
      <td>12002</td>
      <td>74453</td>
      <td>2001</td>
      <td>2</td>
      <td>D7</td>
      <td>17.893048</td>
      <td>2017-12-27</td>
      <td>2030</td>
      <td>1550500</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1</td>
      <td>LOWER EAST SIDE</td>
      <td>08 RENTALS - ELEVATOR APARTMENTS</td>
      <td>2</td>
      <td>343</td>
      <td>68</td>
      <td></td>
      <td>D6</td>
      <td>208 DELANCEY STREET</td>
      <td></td>
      <td>...</td>
      <td>70</td>
      <td>10925</td>
      <td>67296</td>
      <td>2013</td>
      <td>2</td>
      <td>D6</td>
      <td>17.197818</td>
      <td>2017-12-15</td>
      <td>2030</td>
      <td>1550500</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1</td>
      <td>LOWER EAST SIDE</td>
      <td>08 RENTALS - ELEVATOR APARTMENTS</td>
      <td>2</td>
      <td>350</td>
      <td>69</td>
      <td></td>
      <td>D1</td>
      <td>155 ATTORNEY STREET</td>
      <td></td>
      <td>...</td>
      <td>37</td>
      <td>7500</td>
      <td>31455</td>
      <td>2015</td>
      <td>2</td>
      <td>D1</td>
      <td>17.437181</td>
      <td>2018-03-23</td>
      <td>2030</td>
      <td>1550500</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1</td>
      <td>CHINATOWN</td>
      <td>07 RENTALS - WALKUP APARTMENTS</td>
      <td>2</td>
      <td>277</td>
      <td>2</td>
      <td></td>
      <td>C7</td>
      <td>77-79 MADISON STREET</td>
      <td></td>
      <td>...</td>
      <td>30</td>
      <td>5029</td>
      <td>21274</td>
      <td>1918</td>
      <td>2</td>
      <td>C7</td>
      <td>16.300417</td>
      <td>2018-04-25</td>
      <td>2030</td>
      <td>1550500</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1</td>
      <td>LOWER EAST SIDE</td>
      <td>21 OFFICE BUILDINGS</td>
      <td>4</td>
      <td>424</td>
      <td>6</td>
      <td></td>
      <td>O6</td>
      <td>161 BOWERY</td>
      <td></td>
      <td>...</td>
      <td>7</td>
      <td>2664</td>
      <td>18994</td>
      <td>1920</td>
      <td>4</td>
      <td>O6</td>
      <td>17.003927</td>
      <td>2018-08-30</td>
      <td>2030</td>
      <td>1550500</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1</td>
      <td>LOWER EAST SIDE</td>
      <td>07 RENTALS - WALKUP APARTMENTS</td>
      <td>2</td>
      <td>411</td>
      <td>42</td>
      <td></td>
      <td>C7</td>
      <td>138-140 LUDLOW STREET</td>
      <td></td>
      <td>...</td>
      <td>29</td>
      <td>4235</td>
      <td>18180</td>
      <td>1920</td>
      <td>2</td>
      <td>C7</td>
      <td>16.754673</td>
      <td>2018-01-16</td>
      <td>2030</td>
      <td>1550500</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1</td>
      <td>CHINATOWN</td>
      <td>07 RENTALS - WALKUP APARTMENTS</td>
      <td>2</td>
      <td>280</td>
      <td>10</td>
      <td></td>
      <td>C7</td>
      <td>41 HENRY STREET</td>
      <td></td>
      <td>...</td>
      <td>21</td>
      <td>2725</td>
      <td>10860</td>
      <td>1900</td>
      <td>2</td>
      <td>C7</td>
      <td>15.761421</td>
      <td>2018-09-21</td>
      <td>2030</td>
      <td>1550500</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>836</th>
      <td>1</td>
      <td>GREENWICH VILLAGE-CENTRAL</td>
      <td>23 LOFT BUILDINGS</td>
      <td>4</td>
      <td>529</td>
      <td>62</td>
      <td></td>
      <td>L3</td>
      <td>43 BLEECKER STREET</td>
      <td></td>
      <td>...</td>
      <td>15</td>
      <td>6702</td>
      <td>37993</td>
      <td>1900</td>
      <td>4</td>
      <td>L3</td>
      <td>17.526029</td>
      <td>2018-04-23</td>
      <td>2262</td>
      <td>2995000</td>
    </tr>
    <tr>
      <th>837</th>
      <td>1</td>
      <td>SOHO</td>
      <td>08 RENTALS - ELEVATOR APARTMENTS</td>
      <td>2</td>
      <td>488</td>
      <td>8</td>
      <td></td>
      <td>D1</td>
      <td>68 THOMPSON STREET</td>
      <td></td>
      <td>...</td>
      <td>41</td>
      <td>4417</td>
      <td>16749</td>
      <td>1901</td>
      <td>2</td>
      <td>D1</td>
      <td>17.942645</td>
      <td>2017-12-28</td>
      <td>2262</td>
      <td>2995000</td>
    </tr>
    <tr>
      <th>838</th>
      <td>1</td>
      <td>LITTLE ITALY</td>
      <td>07 RENTALS - WALKUP APARTMENTS</td>
      <td>2</td>
      <td>507</td>
      <td>10</td>
      <td></td>
      <td>C7</td>
      <td>244-246 ELIZABETH STREET</td>
      <td></td>
      <td>...</td>
      <td>20</td>
      <td>4027</td>
      <td>14275</td>
      <td>1900</td>
      <td>2</td>
      <td>C7</td>
      <td>14.347439</td>
      <td>2018-06-08</td>
      <td>2262</td>
      <td>2995000</td>
    </tr>
    <tr>
      <th>839</th>
      <td>1</td>
      <td>LITTLE ITALY</td>
      <td>07 RENTALS - WALKUP APARTMENTS</td>
      <td>2</td>
      <td>507</td>
      <td>1</td>
      <td></td>
      <td>C7</td>
      <td>13 PRINCE STREET</td>
      <td></td>
      <td>...</td>
      <td>23</td>
      <td>2918</td>
      <td>12528</td>
      <td>1910</td>
      <td>2</td>
      <td>C7</td>
      <td>16.972511</td>
      <td>2018-03-19</td>
      <td>2262</td>
      <td>2995000</td>
    </tr>
    <tr>
      <th>840</th>
      <td>1</td>
      <td>GREENWICH VILLAGE-CENTRAL</td>
      <td>07 RENTALS - WALKUP APARTMENTS</td>
      <td>2B</td>
      <td>537</td>
      <td>11</td>
      <td></td>
      <td>C7</td>
      <td>228 THOMPSON STREET</td>
      <td></td>
      <td>...</td>
      <td>9</td>
      <td>2476</td>
      <td>8793</td>
      <td>1900</td>
      <td>2</td>
      <td>C7</td>
      <td>16.475770</td>
      <td>2018-07-16</td>
      <td>2262</td>
      <td>2995000</td>
    </tr>
    <tr>
      <th>841</th>
      <td>1</td>
      <td>LITTLE ITALY</td>
      <td>01 ONE FAMILY DWELLINGS</td>
      <td>1</td>
      <td>494</td>
      <td>22</td>
      <td></td>
      <td>A7</td>
      <td>38 PRINCE STREET</td>
      <td></td>
      <td>...</td>
      <td>1</td>
      <td>3309</td>
      <td>8069</td>
      <td>1900</td>
      <td>1</td>
      <td>A7</td>
      <td>16.860033</td>
      <td>2018-08-30</td>
      <td>2262</td>
      <td>2995000</td>
    </tr>
    <tr>
      <th>842</th>
      <td>1</td>
      <td>GREENWICH VILLAGE-CENTRAL</td>
      <td>07 RENTALS - WALKUP APARTMENTS</td>
      <td>2B</td>
      <td>543</td>
      <td>67</td>
      <td></td>
      <td>C5</td>
      <td>133 WEST 3RD   STREET</td>
      <td></td>
      <td>...</td>
      <td>10</td>
      <td>2550</td>
      <td>6158</td>
      <td>1900</td>
      <td>2</td>
      <td>C5</td>
      <td>15.995928</td>
      <td>2017-12-21</td>
      <td>2262</td>
      <td>2995000</td>
    </tr>
    <tr>
      <th>843</th>
      <td>1</td>
      <td>GREENWICH VILLAGE-CENTRAL</td>
      <td>07 RENTALS - WALKUP APARTMENTS</td>
      <td>2A</td>
      <td>531</td>
      <td>39</td>
      <td></td>
      <td>C3</td>
      <td>356 BOWERY</td>
      <td></td>
      <td>...</td>
      <td>4</td>
      <td>1785</td>
      <td>6155</td>
      <td>1910</td>
      <td>2</td>
      <td>C3</td>
      <td>15.761421</td>
      <td>2017-11-22</td>
      <td>2262</td>
      <td>2995000</td>
    </tr>
    <tr>
      <th>844</th>
      <td>1</td>
      <td>SOHO</td>
      <td>22 STORE BUILDINGS</td>
      <td>4</td>
      <td>499</td>
      <td>15</td>
      <td></td>
      <td>K9</td>
      <td>106 PRINCE STREET</td>
      <td></td>
      <td>...</td>
      <td>5</td>
      <td>1000</td>
      <td>5055</td>
      <td>1900</td>
      <td>4</td>
      <td>K9</td>
      <td>18.228309</td>
      <td>2018-02-12</td>
      <td>2262</td>
      <td>2995000</td>
    </tr>
    <tr>
      <th>845</th>
      <td>1</td>
      <td>GREENWICH VILLAGE-CENTRAL</td>
      <td>03 THREE FAMILY DWELLINGS</td>
      <td>1</td>
      <td>526</td>
      <td>45</td>
      <td></td>
      <td>C0</td>
      <td>76 MACDOUGAL STREET</td>
      <td></td>
      <td>...</td>
      <td>3</td>
      <td>2010</td>
      <td>3360</td>
      <td>1899</td>
      <td>1</td>
      <td>C0</td>
      <td>15.882373</td>
      <td>2018-04-09</td>
      <td>2262</td>
      <td>2995000</td>
    </tr>
    <tr>
      <th>846</th>
      <td>1</td>
      <td>GREENWICH VILLAGE-CENTRAL</td>
      <td>01  ONE FAMILY DWELLINGS</td>
      <td>1</td>
      <td>525</td>
      <td>34</td>
      <td></td>
      <td>S1</td>
      <td>168 THOMPSON STREET</td>
      <td></td>
      <td>...</td>
      <td>2</td>
      <td>1350</td>
      <td>5807</td>
      <td>1900</td>
      <td>1</td>
      <td>S1</td>
      <td>16.118096</td>
      <td>2016-11-22</td>
      <td>2262</td>
      <td>2995000</td>
    </tr>
    <tr>
      <th>847</th>
      <td>1</td>
      <td>GREENWICH VILLAGE-CENTRAL</td>
      <td>01  ONE FAMILY DWELLINGS</td>
      <td>1</td>
      <td>526</td>
      <td>51</td>
      <td></td>
      <td>A4</td>
      <td>88 MACDOUGAL STREET</td>
      <td></td>
      <td>...</td>
      <td>1</td>
      <td>2002</td>
      <td>3556</td>
      <td>1899</td>
      <td>1</td>
      <td>A4</td>
      <td>16.341239</td>
      <td>2016-09-09</td>
      <td>2262</td>
      <td>2995000</td>
    </tr>
    <tr>
      <th>848</th>
      <td>1</td>
      <td>GREENWICH VILLAGE-CENTRAL</td>
      <td>02  TWO FAMILY DWELLINGS</td>
      <td>1</td>
      <td>542</td>
      <td>46</td>
      <td></td>
      <td>S2</td>
      <td>109 MACDOUGAL STREET</td>
      <td></td>
      <td>...</td>
      <td>3</td>
      <td>1580</td>
      <td>3902</td>
      <td>1901</td>
      <td>1</td>
      <td>S2</td>
      <td>15.473739</td>
      <td>2017-01-04</td>
      <td>2262</td>
      <td>2995000</td>
    </tr>
    <tr>
      <th>849</th>
      <td>1</td>
      <td>GREENWICH VILLAGE-CENTRAL</td>
      <td>14  RENTALS - 4-10 UNIT</td>
      <td>2A</td>
      <td>526</td>
      <td>61</td>
      <td></td>
      <td>S4</td>
      <td>176 BLEECKER STREET</td>
      <td></td>
      <td>...</td>
      <td>5</td>
      <td>2140</td>
      <td>5040</td>
      <td>1900</td>
      <td>2</td>
      <td>S4</td>
      <td>16.031121</td>
      <td>2017-04-18</td>
      <td>2262</td>
      <td>2995000</td>
    </tr>
    <tr>
      <th>850</th>
      <td>1</td>
      <td>GREENWICH VILLAGE-CENTRAL</td>
      <td>23  LOFT BUILDINGS</td>
      <td>4</td>
      <td>525</td>
      <td>31</td>
      <td></td>
      <td>L9</td>
      <td>124 WEST HOUSTON STREET</td>
      <td></td>
      <td>...</td>
      <td>8</td>
      <td>2500</td>
      <td>13150</td>
      <td>1900</td>
      <td>4</td>
      <td>L9</td>
      <td>16.556351</td>
      <td>2016-05-24</td>
      <td>2262</td>
      <td>2995000</td>
    </tr>
    <tr>
      <th>851</th>
      <td>1</td>
      <td>LITTLE ITALY</td>
      <td>07  RENTALS - WALKUP APARTMENTS</td>
      <td>2</td>
      <td>494</td>
      <td>28</td>
      <td></td>
      <td>C7</td>
      <td>223 MOTT STREET</td>
      <td></td>
      <td>...</td>
      <td>20</td>
      <td>2575</td>
      <td>9170</td>
      <td>1900</td>
      <td>2</td>
      <td>C7</td>
      <td>16.066802</td>
      <td>2017-04-12</td>
      <td>2262</td>
      <td>2995000</td>
    </tr>
    <tr>
      <th>852</th>
      <td>1</td>
      <td>LITTLE ITALY</td>
      <td>07  RENTALS - WALKUP APARTMENTS</td>
      <td>2</td>
      <td>508</td>
      <td>42</td>
      <td></td>
      <td>C7</td>
      <td>239 ELIZABETH STREET</td>
      <td></td>
      <td>...</td>
      <td>11</td>
      <td>1805</td>
      <td>7670</td>
      <td>1914</td>
      <td>2</td>
      <td>C7</td>
      <td>15.761421</td>
      <td>2016-08-09</td>
      <td>2262</td>
      <td>2995000</td>
    </tr>
    <tr>
      <th>853</th>
      <td>1</td>
      <td>LITTLE ITALY</td>
      <td>07  RENTALS - WALKUP APARTMENTS</td>
      <td>2B</td>
      <td>508</td>
      <td>43</td>
      <td></td>
      <td>C7</td>
      <td>237 ELIZABETH STREET</td>
      <td></td>
      <td>...</td>
      <td>8</td>
      <td>1828</td>
      <td>5312</td>
      <td>1910</td>
      <td>2</td>
      <td>C7</td>
      <td>15.796512</td>
      <td>2016-07-11</td>
      <td>2262</td>
      <td>2995000</td>
    </tr>
    <tr>
      <th>854</th>
      <td>1</td>
      <td>LITTLE ITALY</td>
      <td>07  RENTALS - WALKUP APARTMENTS</td>
      <td>2</td>
      <td>510</td>
      <td>26</td>
      <td></td>
      <td>C7</td>
      <td>49-51 PRINCE STREET</td>
      <td></td>
      <td>...</td>
      <td>30</td>
      <td>3850</td>
      <td>16388</td>
      <td>1915</td>
      <td>2</td>
      <td>C7</td>
      <td>15.983707</td>
      <td>2016-08-30</td>
      <td>2262</td>
      <td>2995000</td>
    </tr>
    <tr>
      <th>855</th>
      <td>1</td>
      <td>SOHO</td>
      <td>07  RENTALS - WALKUP APARTMENTS</td>
      <td>2</td>
      <td>489</td>
      <td>36</td>
      <td></td>
      <td>C7</td>
      <td>59 THOMPSON STREET</td>
      <td></td>
      <td>...</td>
      <td>36</td>
      <td>4000</td>
      <td>15918</td>
      <td>1900</td>
      <td>2</td>
      <td>C7</td>
      <td>16.862649</td>
      <td>2016-10-28</td>
      <td>2262</td>
      <td>2995000</td>
    </tr>
    <tr>
      <th>856</th>
      <td>1</td>
      <td>SOHO</td>
      <td>07  RENTALS - WALKUP APARTMENTS</td>
      <td>2B</td>
      <td>496</td>
      <td>35</td>
      <td></td>
      <td>C7</td>
      <td>65 SPRING STREET</td>
      <td></td>
      <td>...</td>
      <td>10</td>
      <td>1927</td>
      <td>7690</td>
      <td>1920</td>
      <td>2</td>
      <td>C7</td>
      <td>16.489659</td>
      <td>2016-11-01</td>
      <td>2262</td>
      <td>2995000</td>
    </tr>
    <tr>
      <th>857</th>
      <td>1</td>
      <td>SOHO</td>
      <td>14  RENTALS - 4-10 UNIT</td>
      <td>2A</td>
      <td>520</td>
      <td>79</td>
      <td></td>
      <td>S5</td>
      <td>51 MAC DOUGAL STREET</td>
      <td></td>
      <td>...</td>
      <td>6</td>
      <td>1937</td>
      <td>5240</td>
      <td>1920</td>
      <td>2</td>
      <td>S5</td>
      <td>16.102982</td>
      <td>2016-12-22</td>
      <td>2262</td>
      <td>2995000</td>
    </tr>
    <tr>
      <th>858</th>
      <td>1</td>
      <td>SOHO</td>
      <td>41  TAX CLASS 4 - OTHER</td>
      <td>4</td>
      <td>511</td>
      <td>19</td>
      <td></td>
      <td>O2</td>
      <td>19 EAST HOUSTON STREET</td>
      <td></td>
      <td>...</td>
      <td>2</td>
      <td>6190</td>
      <td>41267</td>
      <td>2016</td>
      <td>4</td>
      <td>Z7</td>
      <td>17.474575</td>
      <td>2016-05-20</td>
      <td>2262</td>
      <td>2995000</td>
    </tr>
    <tr>
      <th>859</th>
      <td>1</td>
      <td>SOHO</td>
      <td>41  TAX CLASS 4 - OTHER</td>
      <td>4</td>
      <td>513</td>
      <td>28</td>
      <td></td>
      <td>K2</td>
      <td>155 MERCER STREET</td>
      <td></td>
      <td>...</td>
      <td>1</td>
      <td>4600</td>
      <td>18120</td>
      <td>1920</td>
      <td>4</td>
      <td>Z9</td>
      <td>18.348110</td>
      <td>2016-11-04</td>
      <td>2262</td>
      <td>2995000</td>
    </tr>
    <tr>
      <th>860</th>
      <td>1</td>
      <td>FINANCIAL</td>
      <td>08 RENTALS - ELEVATOR APARTMENTS</td>
      <td>2A</td>
      <td>79</td>
      <td>26</td>
      <td></td>
      <td>D5</td>
      <td>142 FULTON STREET</td>
      <td></td>
      <td>...</td>
      <td>6</td>
      <td>2898</td>
      <td>13194</td>
      <td>1909</td>
      <td>2</td>
      <td>D5</td>
      <td>16.867628</td>
      <td>2018-10-17</td>
      <td>1723</td>
      <td>1695000</td>
    </tr>
    <tr>
      <th>861</th>
      <td>1</td>
      <td>SOUTHBRIDGE</td>
      <td>14 RENTALS - 4-10 UNIT</td>
      <td>2A</td>
      <td>90</td>
      <td>23</td>
      <td></td>
      <td>S5</td>
      <td>17 ANN STREET</td>
      <td></td>
      <td>...</td>
      <td>6</td>
      <td>855</td>
      <td>5520</td>
      <td>1900</td>
      <td>2</td>
      <td>S5</td>
      <td>15.573369</td>
      <td>2018-01-09</td>
      <td>1723</td>
      <td>1695000</td>
    </tr>
    <tr>
      <th>862</th>
      <td>1</td>
      <td>FINANCIAL</td>
      <td>26  OTHER HOTELS</td>
      <td>4</td>
      <td>78</td>
      <td>20</td>
      <td></td>
      <td>H8</td>
      <td>106 FULTON STREET</td>
      <td></td>
      <td>...</td>
      <td>3</td>
      <td>5646</td>
      <td>74863</td>
      <td>1900</td>
      <td>4</td>
      <td>H8</td>
      <td>17.749295</td>
      <td>2016-10-07</td>
      <td>1723</td>
      <td>1695000</td>
    </tr>
    <tr>
      <th>863</th>
      <td>1</td>
      <td>SOUTHBRIDGE</td>
      <td>08  RENTALS - ELEVATOR APARTMENTS</td>
      <td>2</td>
      <td>92</td>
      <td>3</td>
      <td></td>
      <td>D5</td>
      <td>165-167 WILLIAM STREET</td>
      <td></td>
      <td>...</td>
      <td>13</td>
      <td>3480</td>
      <td>31073</td>
      <td>1908</td>
      <td>2</td>
      <td>D5</td>
      <td>16.759950</td>
      <td>2017-02-10</td>
      <td>1723</td>
      <td>1695000</td>
    </tr>
    <tr>
      <th>864</th>
      <td>1</td>
      <td>CIVIC CENTER</td>
      <td>14  RENTALS - 4-10 UNIT</td>
      <td>2B</td>
      <td>145</td>
      <td>10</td>
      <td></td>
      <td>S9</td>
      <td>121 CHAMBERS STREET</td>
      <td></td>
      <td>...</td>
      <td>10</td>
      <td>3788</td>
      <td>18500</td>
      <td>1930</td>
      <td>2</td>
      <td>S9</td>
      <td>16.600522</td>
      <td>2016-08-16</td>
      <td>2083</td>
      <td>4075000</td>
    </tr>
    <tr>
      <th>865</th>
      <td>1</td>
      <td>CIVIC CENTER</td>
      <td>23  LOFT BUILDINGS</td>
      <td>4</td>
      <td>136</td>
      <td>20</td>
      <td></td>
      <td>L8</td>
      <td>124 CHAMBERS STREET</td>
      <td></td>
      <td>...</td>
      <td>6</td>
      <td>1894</td>
      <td>10359</td>
      <td>1920</td>
      <td>4</td>
      <td>L8</td>
      <td>15.761421</td>
      <td>2016-06-01</td>
      <td>2083</td>
      <td>4075000</td>
    </tr>
  </tbody>
</table>
<p>866 rows × 23 columns</p>
</div>




```python
train1= features # can change to scaled_features or features to test regresion model with or without categorical values 
labels=target

```


```python
x_train , x_test , y_train , y_test = train_test_split(train1 , labels , test_size = 0.20,random_state =30)
```


```python
lm = LinearRegression()
lm.fit(x_train,y_train)

# evaluation using r-square

lm.score(x_train,y_train)
# x_test
```




    0.7317281102976905



### We create a scatterplot between the predicted prices, (where m is the fitted model) and the original prices. 

# A perfect model would get us a scatterplot where all the data lies on the 45 degree line. 

# Data  shows we are more accurate when we hit prices around 160 million


```python
import matplotlib.pyplot as plt
```


```python
predicted_prices = m7.fittedvalues

plt.scatter(predicted_prices, new_df.SALE_PRICE)
plt.xlabel("Predicted Prices by Model")
plt.ylabel='Original Prices'
plt.title='Predictions vs. Original Prices'
plt.xlim((10,25))
plt.show()


```


![png](PredictAPrice%20-%20Notebook_files/PredictAPrice%20-%20Notebook_41_0.png)



```python
x = m7.fittedvalues
y = m7.resid
plt.scatter(x, y)

plt.xlabel("Fitted Values")
# plt.ylabel("Residual")
# plt.title("Fitted Values vs. Residuals")

## the model is predicting heteroskedastically, 
because we are overpredicting the price when the actual price is low and underpredicting when it is high
```




    Text(0.5,0,'Fitted Values')




![png](PredictAPrice%20-%20Notebook_files/PredictAPrice%20-%20Notebook_42_1.png)



```python

```


```python
sm.qqplot(m7.resid,line='45')
```




![png](PredictAPrice%20-%20Notebook_files/PredictAPrice%20-%20Notebook_44_0.png)




![png](PredictAPrice%20-%20Notebook_files/PredictAPrice%20-%20Notebook_44_1.png)



```python
np.sqrt(metrics.mean_squared_error(y_test, y_pred))/np.std(y_train)
```




    SALE_PRICE    0.502126
    dtype: float64




```python
reg.fit(x_train,y_train)
reg.score(x_test,y_test) 
#highest score with all variables (2110) is .79
```




    0.7250078942974605




```python
y_pred = lm.predict(x_test) #from seans ridge nad lasso slides 

print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

```

    MSE: 184264415698494.62
    RMSE: 13574402.959191047


## Overall Metrics

* Root Mean Square Error : 13574402.959191047


```python
from sklearn.metrics import median_absolute_error
median_absolute_error(y_test, y_pred)
# sklearn.metrics.median_absolute_error(y_true, y_pred)[source]
```




    1585151.4366711155



## Next Steps

* Regularization (with Lasso and Ridge)
* Determine why our predictions are heteroskedastic.


```python
coef = pd.DataFrame(data=lm.coef_, columns=x_train.columns ) #takes co-effficent and pairs up with columns, and looks at 

model_coef = coef.T.sort_values(by=0).T

model_coef.plot(kind='bar', title='Modal Coefficients', legend=False, figsize=(16,8))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c24c92b00>




![png](PredictAPrice%20-%20Notebook_files/PredictAPrice%20-%20Notebook_51_1.png)



```python
y_test.std()
```




    SALE_PRICE    2.596045e+07
    dtype: float64




```python
X_train=x_train #***
X_test=x_test
ridgeReg = Ridge(alpha=.50, normalize=True)

ridgeReg.fit(X_train,y_train)

y_pred = ridgeReg.predict(X_test)

#calculating mse

print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred))/ y_test.std())
coef = pd.DataFrame(data=ridgeReg.coef_, columns=X_train.columns )

model_coef = coef.T.sort_values(by=0).T

model_coef.plot(kind='bar', title='RR Model Coefficients', legend=False, figsize=(16,8))
```

    MSE: 247987123402372.03
    RMSE: 15747606.91033314
    SALE_PRICE    0.6066
    dtype: float64





    <matplotlib.axes._subplots.AxesSubplot at 0x1c260557f0>




![png](PredictAPrice%20-%20Notebook_files/PredictAPrice%20-%20Notebook_53_2.png)



```python
#Identifing Outliers
X_train=x_train #***
X_test=x_test
ridgeReg = Ridge(alpha=.20, normalize=True)

ridgeReg.fit(X_train,y_train)

y_pred = ridgeReg.predict(X_test)

#calculating mse

print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred))/ y_test.std())
coef = pd.DataFrame(data=ridgeReg.coef_, columns=X_train.columns )

model_coef = coef.T.sort_values(by=0).T

model_coef.plot(kind='bar', title='RR Model Coefficients', legend=False, figsize=(16,8))
```

    MSE: 218642713567700.6
    RMSE: 14786572.069540005
    SALE_PRICE    0.569581
    dtype: float64





    <matplotlib.axes._subplots.AxesSubplot at 0x1c2617dc50>




![png](PredictAPrice%20-%20Notebook_files/PredictAPrice%20-%20Notebook_54_2.png)



```python
plt.scatter(-0, 25, s=10000, alpha=0.3, c = 'r' )
plt.scatter(45, 21, s=8000, alpha=0.3, c = 'r' )


predicted_prices = m7.fittedvalues

plt.xlabel("Predicted Prices by Model")
# plt.ylabel()
# plt.title("Predictions vs. Original Prices")
plt.scatter(predicted_prices, new_df.SALE_PRICE)

```




    <matplotlib.collections.PathCollection at 0x1c26e057f0>




![png](PredictAPrice%20-%20Notebook_files/PredictAPrice%20-%20Notebook_55_1.png)



```python
#***Lasso regression not only helps in reducing over-fitting but it can help us in feature selection.
from sklearn.linear_model import Lasso


lassoReg = Lasso(alpha=50, normalize=True)

lassoReg.fit(X_train,y_train)

y_pred = lassoReg.predict(X_test)

#calculating mse

print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred))/ y_test.std())



coef = pd.DataFrame(data=lassoReg.coef_, index=X_train.columns )
model_coef = coef.sort_values(by=0).T

model_coef.plot(kind='bar', title='Lasso Model Coefficients', legend=False, figsize=(16,8))

```

    /Users/chrischung/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)


    MSE: 184445775930621.6
    RMSE: 13581081.544951476
    SALE_PRICE    0.523145
    dtype: float64





    <matplotlib.axes._subplots.AxesSubplot at 0x1c262c1828>




![png](PredictAPrice%20-%20Notebook_files/PredictAPrice%20-%20Notebook_56_3.png)



```python
from sklearn.feature_selection import RFE
rfe = RFE(lm, n_features_to_select=10)
rfe.fit(features_selected_train,y_train)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-35-22d8e31e57c9> in <module>()
          1 from sklearn.feature_selection import RFE
          2 rfe = RFE(lm, n_features_to_select=10)
    ----> 3 rfe.fit(features_selected_train,y_train)
    

    NameError: name 'features_selected_train' is not defined



```python

plt.style.use('fivethirtyeight')
plt.figure(figsize=(12,8))
sns.distplot(new_df.SALE_PRICE, bins = 25)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,1))
plt.xlabel("House Sales Price in USD")
plt.ylabel("Number of Houses")
plt.title("House Sales Price Distribution")

```


```python
# x_test
```


```python
from sklearn import preprocessing
from sklearn import pipeline

scaler = preprocessing.StandardScaler()
X_test=x_test#[['RESIDENTIAL_UNITS', 'COMMERCIAL_UNITS', 'LAND_SQUARE_FEET','GROSS_SQUARE_FEET', 'YEAR_BUILT']]
X_test1=x_test#[['RESIDENTIAL_UNITS', 'COMMERCIAL_UNITS', 'LAND_SQUARE_FEET','GROSS_SQUARE_FEET', 'YEAR_BUILT',"BUILDING_CLASS_CATEGORY_01ONEFAMILYDWELLINGS"]]
X_test1=x_test[['RESIDENTIAL_UNITS', 'COMMERCIAL_UNITS', 'LAND_SQUARE_FEET','GROSS_SQUARE_FEET', 'YEAR_BUILT',"BUILDING_CLASS_CATEGORY_01ONEFAMILYDWELLINGS"]]
X_train=x_train
```


```python
scaler.fit(features.iloc[:,:-1])

```


```python
len(X_test1.columns[:-1])
len(X_test1.iloc[:,:-1])
X_test1.columns.shape
```


```python
scaler.fit(X_train.iloc[:,:-1])
features_scaled_train = pd.DataFrame(scaler.transform(X_train.iloc[:,:-1]), columns=X_train.columns[:-1], index=X_train.index)

features_scaled_train.head()
```


```python
features_scaled_test = pd.DataFrame(scaler.transform(X_test.iloc[:,:-1]), columns=X_test.columns[:-1], index=X_test.index)

features_scaled_test.head()
```


```python
poly = preprocessing.PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
features_64_train = pd.DataFrame(poly.fit_transform(features_scaled_train), columns=poly.get_feature_names(features_scaled_train.columns))
features_64_train.head()

```


```python
pd.set_option('display.max_columns', 100)
features_64_train.head()
features_64_test = pd.DataFrame(poly.fit_transform(features_scaled_test), columns=poly.get_feature_names(features_scaled_test.columns))
features_64_test.head()
```


```python
***
```


```python
from sklearn.feature_selection import VarianceThreshold
thresholder = VarianceThreshold(threshold=.5)

def variance_threshold_selector(data, threshold=0.5):
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]
```


```python
features_selected_train = variance_threshold_selector(features_64_train)
# features_selected_train = variance_threshold_selector(features_64_train)
```


```python
features_selected_train.head()
```


```python
import seaborn as sns

sns.set(style="white")


# Compute the correlation matrix
corr = features_selected_train.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
```


```python
# Create correlation matrix
corr_matrix = features_selected_train.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
```


```python
upper
```


```python
features_selected_train.drop(columns=to_drop, inplace=True)
```


```python
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression, mutual_info_regression
```


```python
def information_selector(X, y, scoring, k=5):
    selector = SelectKBest(score_func=scoring, k=k)
    selector.fit(X, y)
    return X[X.columns[selector.get_support(indices=True)]]
test = SelectKBest(score_func=mutual_info_regression, k=30)
fit = test.fit(features_selected_train, y_train)
```


```python
features_selected_train[features_selected_train.columns[fit.get_support(indices=True)]].head()
```


```python
features_selected_train = information_selector(features_selected_train, y_train, mutual_info_regression, k=30)
```


```python
# fit a model
lm = linear_model.LinearRegression()
model = lm.fit(features_selected_train, y_train)
```


```python
features_selected_test = features_64_test[features_selected_train.columns]
y_pred = lm.predict(features_selected_test)

print(metrics.mean_absolute_error(y_test, y_pred))
print(metrics.mean_squared_error(y_test, y_pred))
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
```


```python

```


```python
from sklearn.feature_selection import RFE
rfe = RFE(lm, n_features_to_select=10)
rfe.fit(features_selected_train,y_train)
```


```python
def ranking(ranks, names, order=1):

    ranks = map(lambda x: (x,2), ranks)
    return list(sorted(zip(ranks, names),reverse=True))
```


```python
rankings = ranking(np.abs(lm.coef_), features_selected_train.columns)
```


```python
rankings[:15]
```


```python
[item[1] for item in rankings[0:15]]
```


```python
final_columns = [item[1] for item in rankings[0:15]]

```


```python
lm = linear_model.LinearRegression()
model = lm.fit(features_selected_train[final_columns], y_train)
```


```python
features_selected_test = features_64_test[final_columns]
y_pred = lm.predict(features_selected_test)

print(metrics.mean_absolute_error(y_test, y_pred))
print(metrics.mean_squared_error(y_test, y_pred))
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
```


```python
# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
```


```python
### Module 2 Projects

Projects are designed to review the material we covered in Module 2:

* cleaning data with numpy and pandas
* probability and combinatorics
* probability distributions
* hypothesis testing
* simple linear regression
* multiple linear regression
* cross validation and the bias/variance tradeoff

Ask a main question with which you can use a regression to answer. The other topics we learned in Module 2 can be used as further justification for your answers to subsequent questions.

Sample Questions:

* What best determines the final auction price of an item?
* What are the key factors in determining a country's happiness level?
* Is there a way we can predict the spread of a football game?


### Data
* You must have at least 4 different features in your models (independent variables) with at least one target (dependent variable).
* Your data must contain at least one categorical feature and at least one numerical feature
* **BONUS**: Challenge yourself to obtain a unique dataset (either from webscraping or querying APIs)

### The Deliverables

1. ** A well documented Jupyter Notebook** containing any code you've written for this project, comments explaining it, and graphical visualizations.

## Requirements

#### Organization/Code Cleanliness

* The notebook should be well organized, easy to follow,  and code should be commented where appropriate.  
    * Level Up: The notebook contains well-formatted, professional looking markdown cells explaining any substantial code. All functions have docstrings that act as professional-quality documentation
* The notebook is written for a technical audiences with a way to both understand your approach and reproduce your results. The target audience for this deliverable is other data scientists looking to validate your findings.

#### Visualizations & EDA (Exploratory Data Analysis)

* Your project contains at least 4 _meaningful_ data visualizations, with corresponding interpretations. All visualizations are well labeled with axes labels, a title, and a legend (when appropriate)  
* You pose at least 3 meaningful questions and answer them through EDA.  These questions should be well labeled and easy to identify inside the notebook.
    * **Level Up**: Each question is clearly answered with a visualization that makes the answer easy to understand.   
* Your notebook should contain 1 - 2 paragraphs briefly explaining your approach to this project.


#### Model Quality/Approach

* Your model should not include any predictors with p-values greater than .05 (unless you can justify)
* Your model should have cross-validation and account for the bias-variance tradeoff  
* Your notebook shows an iterative approach to modeling, and details the parameters and results of the model at each iteration.  
    * **Level Up**: Whenever necessary, you briefly explain the changes made from one iteration to the next, and why you made these choices.  
* You provide at least 1 paragraph explaining your final model.   
```


```python

```
