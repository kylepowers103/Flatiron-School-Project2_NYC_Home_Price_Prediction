
# PredictAPrice

## As a property seller in Manhattan, what are factors that affect what you should price at?

### Datasets

We used NYC.gov's Rolling Sales data from the last 12 months, and 2 Zillow datasets: one showing all median prices and one of all square footage of properties sold. Both are in the year 2018. We merged all datasets together into a main dataframe.

### Data Cleaning

We filtered out all 0 value rows and extreme outliers.

### Transforming Data

When we plotted out our graphs, the normal distribution was postively skewed. We log transformed the data and the graph read much better.




## Access and Filter necessary info from 3 Real Estate datasets


```python
#combined Rolling Sales Manhattan excel sheets for years 2016-2018
data=pd.read_excel("rollingsales_manhattan.xls", skiprows=1) 

##Filter 0 values and very extreme outliers 
data = data[data['SALE_PRICE'] > 100] 
data = data[data['SALE_PRICE'] < 250000000]  
data=data[data['GROSS SQUARE FEET'] > 100] 

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

    skewness: 4.750264
    kurtosis: 26.823504

![png](PredictAPrice%20-%20Notebook_files/PredictAPrice%20-%20Notebook_13_2.png)


![png](PredictAPrice%20-%20Notebook_files/PredictAPrice%20-%20Notebook_13_3.png)


## Transform Data


```python

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

![png](PredictAPrice%20-%20Notebook_files/PredictAPrice%20-%20Notebook_15_1.png)



![png](PredictAPrice%20-%20Notebook_files/PredictAPrice%20-%20Notebook_15_2.png)


# # Step 1: Checking for Linearity  using Scatterplots


```python

sns.regplot(y=new_df.SALE_PRICE, x=new_df['RESIDENTIAL_UNITS'], data=new_df, fit_reg = True)

```




![png](PredictAPrice%20-%20Notebook_files/PredictAPrice%20-%20Notebook_17_2.png)



```python
sns.regplot(y=new_df.SALE_PRICE, x=new_df['COMMERCIAL_UNITS'], data=new_df, fit_reg = True)


```




![png](PredictAPrice%20-%20Notebook_files/PredictAPrice%20-%20Notebook_18_2.png)



```python
sns.regplot(y=new_df.SALE_PRICE, x=new_df['LAND_SQUARE_FEET'], data=new_df, fit_reg = True)
```

![png](PredictAPrice%20-%20Notebook_files/PredictAPrice%20-%20Notebook_19_2.png)



```python
sns.regplot(y=new_df.SALE_PRICE, x=new_df['GROSS_SQUARE_FEET'], data=new_df, fit_reg = True)

```


![png](PredictAPrice%20-%20Notebook_files/PredictAPrice%20-%20Notebook_20_2.png)


# Graph showing the distribution of prices by neighborhoood


```python

plt.figure(figsize=(20,7))
sns.stripplot(x = new_df.NEIGHBORHOOD, y = new_df.SALE_PRICE,
              order = np.sort(new_df.NEIGHBORHOOD.unique()),
              jitter=0.1, alpha=0.5)
plt.xticks(rotation=90)



```

![png](PredictAPrice%20-%20Notebook_files/PredictAPrice%20-%20Notebook_22_1.png)


# Distribution of square footage and sale price


```python

plt.figure(figsize=(12,7))
sns.stripplot(x = new_df.GROSS_SQUARE_FEET, y = new_df.SALE_PRICE,
              order = np.sort(new_df.GROSS_SQUARE_FEET),
              jitter=0.1, alpha=0.5)
plt.xticks(rotation=45)
```




   

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
m5 = ols('SALE_PRICE ~GROSS_SQUARE_FEET ',new_df).fit()
print(m1.summary())

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
```




    0.7317281102976905



# We create a scatterplot between the predicted prices, (where m is the fitted model) and the original prices. 

## A perfect model would get us a scatterplot where all the data lies on the 45 degree line. 


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
