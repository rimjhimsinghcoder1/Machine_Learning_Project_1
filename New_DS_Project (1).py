```python
import pandas as pd

# Replace the file path with the correct path on your machine
file_path = r'C:\Users\Rimjhim\Desktop\HDI_1.csv'

# Import the CSV file
df = pd.read_csv(file_path)

# Display the shape of the DataFrame
print(df.shape)

```

    (191, 9)
    


```python

```


```python
# Check the data types of each column
print(df.dtypes)
```

    HDI_rank                                int64
    Country                                object
    HDI                                   float64
    LE                                    float64
    Expected_years_of_schooling           float64
    Mean_years_of_schooling               float64
    GNI_per_capita                        float64
    GNI per capita rank minus HDI rank      int64
    HDI_rank_2020                           int64
    dtype: object
    


```python
# Convert specified columns to integer
columns_to_convert = ['HDI', 'LE', 'Expected_years_of_schooling', 'Mean_years_of_schooling', 'GNI_per_capita']
df[columns_to_convert] = df[columns_to_convert].astype(int)

```


```python
# Check the data types of each column
print(df.dtypes)
```

    HDI_rank                               int64
    Country                               object
    HDI                                    int32
    LE                                     int32
    Expected_years_of_schooling            int32
    Mean_years_of_schooling                int32
    GNI_per_capita                         int32
    GNI per capita rank minus HDI rank     int64
    HDI_rank_2020                          int64
    dtype: object
    


```python
import statsmodels.api as sm

# Assuming df is your DataFrame
X = df[['LE', 'Mean_years_of_schooling']]
y = df['GNI_per_capita']

# Add a constant term to the independent variables matrix
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Get the summary of the regression
print(model.summary())


```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:         GNI_per_capita   R-squared:                       0.569
    Model:                            OLS   Adj. R-squared:                  0.564
    Method:                 Least Squares   F-statistic:                     124.1
    Date:                Wed, 13 Dec 2023   Prob (F-statistic):           4.33e-35
    Time:                        13:48:44   Log-Likelihood:                -2098.4
    No. Observations:                 191   AIC:                             4203.
    Df Residuals:                     188   BIC:                             4213.
    Df Model:                           2                                         
    Covariance Type:            nonrobust                                         
    ===========================================================================================
                                  coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------------------
    const                   -1.069e+05   1.16e+04     -9.248      0.000    -1.3e+05   -8.41e+04
    LE                       1588.1299    200.457      7.923      0.000    1192.696    1983.564
    Mean_years_of_schooling  1719.0068    488.347      3.520      0.001     755.663    2682.350
    ==============================================================================
    Omnibus:                      136.040   Durbin-Watson:                   1.195
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1593.709
    Skew:                           2.546   Prob(JB):                         0.00
    Kurtosis:                      16.203   Cond. No.                         797.
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    


```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Assuming df_clean is your DataFrame
x_data = df['LE']
y_data = df['GNI_per_capita']

# Define the exponential function
def exponential_func(x, m):
    return np.exp(m * x)

# Fit the curve
params, covariance = curve_fit(exponential_func, x_data, y_data)

# Get the fitted values
y_fit = exponential_func(x_data, *params)

# Scatter plot
plt.scatter(x_data, y_data, label='Data points')

# Plot the fitted curve
plt.plot(x_data, y_fit, color='red', label='Exponential fit')

plt.title('Exponential Fit to Data')
plt.xlabel('Life Expectancy (LE)')
plt.ylabel('Gross National Income (GNI) per capita')
plt.legend()
plt.show()
```


    
![png](output_6_0.png)
    



```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Assuming df_clean is your DataFrame
x_data = df['LE']
y_data = df['GNI_per_capita']

# Add jitter to the data
jitter = 0.2  # Adjust the amount of jitter as needed
x_jittered = x_data + np.random.uniform(-jitter, jitter, len(x_data))
y_jittered = y_data + np.random.uniform(-jitter, jitter, len(y_data))

# Define the exponential function
def exponential_func(x, m):
    return np.exp(m * x)

# Fit the curve
params, covariance = curve_fit(exponential_func, x_jittered, y_jittered)

# Get the fitted values
y_fit = exponential_func(x_data, *params)

# Scatter plot with jitter
plt.scatter(x_jittered, y_jittered, label='Data points with jitter', alpha=0.7)

# Plot the fitted curve
plt.plot(x_data, y_fit, color='red', label='Exponential fit')

plt.title('Exponential Fit to Data with Jitter')
plt.xlabel('Life Expectancy (LE)')
plt.ylabel('Gross National Income (GNI) per capita')
plt.legend()
plt.show()

```


    
![png](output_7_0.png)
    



```python
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming df is your DataFrame with multiple columns
# Selecting a subset of columns for the example
selected_columns = ['HDI', 'LE', 'Expected_years_of_schooling', 'Mean_years_of_schooling', 'GNI_per_capita']

# Plotting a pair plot for selected columns
sns.pairplot(df[selected_columns])
plt.show()

```

    C:\Users\Rimjhim\anaconda3\Lib\site-packages\seaborn\axisgrid.py:118: UserWarning: The figure layout has changed to tight
      self._figure.tight_layout(*args, **kwargs)
    


    
![png](output_8_1.png)
    



```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

# Assuming df is your DataFrame
features = ['HDI', 'LE', 'Expected_years_of_schooling', 'Mean_years_of_schooling', 'GNI_per_capita']

# Separate features
x = df[features]

# Standardize the features
x_standardized = StandardScaler().fit_transform(x)

# Apply PCA
pca = PCA(n_components=3)
principal_components = pca.fit_transform(x_standardized)

# Create a DataFrame with the first three principal components
pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])

# Plot the 3D scatter plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pc_df['PC1'], pc_df['PC2'], pc_df['PC3'], c='b', marker='o')

ax.set_xlabel('Principal Component 5')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 4')
ax.set_title('3D Scatter Plot of Principal Components')

plt.show()

```


    
![png](output_9_0.png)
    



```python
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Assuming df is your DataFrame
features = ['HDI', 'LE', 'Expected_years_of_schooling', 'Mean_years_of_schooling', 'GNI_per_capita']

# Separate features
x = df[features]

# Standardize the features
x_standardized = StandardScaler().fit_transform(x)

# Apply PCA
pca = PCA()
principal_components = pca.fit_transform(x_standardized)

# Create a DataFrame with the principal components
pc_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(len(features))])

# Variance explained by each principal component
explained_variance_ratio = pca.explained_variance_ratio_

# Plot the explained variance ratio
plt.bar(range(1, len(features) + 1), explained_variance_ratio, alpha=0.5, align='center')
plt.xlabel('Principal Component Number')
plt.ylabel('Variance Explained Ratio')
plt.show()

```


    
![png](output_10_0.png)
    



```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Assuming df is your DataFrame
features = ['HDI', 'LE', 'Expected_years_of_schooling', 'Mean_years_of_schooling', 'GNI_per_capita']

# Separate features
x = df[features]

# Standardize the features
x_standardized = StandardScaler().fit_transform(x)

# Apply PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(x_standardized)

# Create a DataFrame with the first two principal components
pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Plot the 2D scatter plot
plt.figure(figsize=(8, 8))
plt.scatter(pc_df['PC1'], pc_df['PC2'], c='b', marker='o')
plt.title('2D Scatter Plot of Principal Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

```


    
![png](output_11_0.png)
    



```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Assuming df is your DataFrame
features = ['HDI', 'LE', 'Expected_years_of_schooling', 'Mean_years_of_schooling', 'GNI_per_capita']

# Separate features
x = df[features]

# Standardize the features
x_standardized = StandardScaler().fit_transform(x)

# Apply PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(x_standardized)

# Create a DataFrame with the first two principal components
pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Variance explained by each principal component
explained_variance_ratio = pca.explained_variance_ratio_

# Cumulative variance explained
cumulative_variance_ratio = explained_variance_ratio.cumsum()

# Loadings (coefficients of the original features in the principal components)
loadings = pca.components_.T * (explained_variance_ratio ** 0.5)

# Plot the 2D scatter plot
plt.figure(figsize=(8, 8))
plt.scatter(pc_df['PC1'], pc_df['PC2'], c='b', marker='o')
plt.title('2D Scatter Plot of Principal Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Print results
print("Explained Variance Ratio:")
print(explained_variance_ratio)
print("\nCumulative Variance Explained:")
print(cumulative_variance_ratio)
print("\nLoadings:")
print(loadings)

```


    
![png](output_12_0.png)
    


    Explained Variance Ratio:
    [0.79082318 0.09921349]
    
    Cumulative Variance Explained:
    [0.79082318 0.89003667]
    
    Loadings:
    [[ 0.         -0.        ]
     [ 0.45728067  0.03094578]
     [ 0.45163818 -0.13547108]
     [ 0.44501867 -0.13227551]
     [ 0.42390908  0.24981318]]
    


```python
import seaborn as sns
import matplotlib.pyplot as plt

# Set the style for the plots
sns.set(style="whitegrid")

# Plot histograms for each feature in the DataFrame
plt.figure(figsize=(16, 10))
for i, feature in enumerate(features[1:]):  # Exclude 'HDI' for better visualization
    plt.subplot(2, 2, i + 1)
    sns.histplot(df[feature], kde=True, color='skyblue')
    plt.title(f'Histogram of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


```


    
![png](output_13_0.png)
    



```python

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Assuming df is your DataFrame
features = ['HDI', 'LE', 'Expected_years_of_schooling', 'Mean_years_of_schooling', 'GNI_per_capita']

# Choose a range of indices or specify a list of countries
start_index = 0  # Adjust the starting index
end_index = 160  # Adjust the ending index (exclusive)

# Filter DataFrame for the specified range of countries
subset_df = df.iloc[start_index:end_index]

# Separate features
x = subset_df[features]

# Standardize the features
x_standardized = StandardScaler().fit_transform(x)

# Apply PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(x_standardized)

# Create a DataFrame with the first two principal components
pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pc_df['Country'] = subset_df['Country']  # Add country names to the DataFrame

# Plot the 2D scatter plot with country names
plt.figure(figsize=(10, 8))
plt.scatter(pc_df['PC1'], pc_df['PC2'], c='b', marker='o')
for i, txt in enumerate(pc_df['Country']):
    plt.annotate(txt, (pc_df['PC1'][i], pc_df['PC2'][i]), fontsize=8)

plt.title('2D Scatter Plot of Principal Components with Country Names (Subset)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


```

    C:\Users\Rimjhim\anaconda3\Lib\site-packages\IPython\core\pylabtools.py:152: UserWarning: Glyph 129 (\x81) missing from current font.
      fig.canvas.print_figure(bytes_io, **kw)
    


    
![png](output_14_1.png)
    



```python
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming df is your DataFrame
heatmap_data = df[['HDI_rank',  'LE', 'Expected_years_of_schooling', 'Mean_years_of_schooling', 'GNI_per_capita']]

plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()
```


    
![png](output_15_0.png)
    



```python
import geopandas as gpd
import matplotlib.pyplot as plt

# Load world map shapefile
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Merge world map with your data (assuming 'Country' is a common column)
merged = world.merge(df, how='left', left_on='name', right_on='Country')

# Plot the map
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# You can choose either HDI or HDI_rank to color the map
variable_to_plot = 'HDI_rank'  # Change this to 'HDI_rank' if needed

# Plot the map with colors based on the chosen variable
merged.plot(column=variable_to_plot, cmap='YlOrRd', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)

# Add a title
plt.title(f'World Map - {variable_to_plot}', fontsize=16)

# Display the plot
plt.show()
```

    C:\Users\Rimjhim\AppData\Local\Temp\ipykernel_30108\2364781036.py:5: FutureWarning: The geopandas.dataset module is deprecated and will be removed in GeoPandas 1.0. You can get the original 'naturalearth_lowres' data from https://www.naturalearthdata.com/downloads/110m-cultural-vectors/.
      world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    


    
![png](output_16_1.png)
    



```python
import geopandas as gpd
import matplotlib.pyplot as plt

# Load world map shapefile
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Merge world map with your data (assuming 'Country' is a common column)
merged = world.merge(df, how='left', left_on='name', right_on='Country')

# Plot the map
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# You can choose either HDI or HDI_rank to color the map
variable_to_plot = 'HDI_rank'  # Change this to 'HDI_rank' if needed

# Plot the map with colors based on the chosen variable
merged.plot(column=variable_to_plot, cmap='YlOrRd', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)

# Add continent names to the map
for x, y, label in zip(world.geometry.centroid.x, world.geometry.centroid.y, world['continent']):
    ax.text(x, y, label, fontsize=8, ha='center', va='center')

# Add a title
plt.title(f'World Map - {variable_to_plot}', fontsize=16)

# Display the plot
plt.show()

```

    C:\Users\Rimjhim\AppData\Local\Temp\ipykernel_30108\4024372253.py:5: FutureWarning: The geopandas.dataset module is deprecated and will be removed in GeoPandas 1.0. You can get the original 'naturalearth_lowres' data from https://www.naturalearthdata.com/downloads/110m-cultural-vectors/.
      world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    C:\Users\Rimjhim\AppData\Local\Temp\ipykernel_30108\4024372253.py:20: UserWarning: Geometry is in a geographic CRS. Results from 'centroid' are likely incorrect. Use 'GeoSeries.to_crs()' to re-project geometries to a projected CRS before this operation.
    
      for x, y, label in zip(world.geometry.centroid.x, world.geometry.centroid.y, world['continent']):
    


    
![png](output_17_1.png)
    



```python


```


```python

```
