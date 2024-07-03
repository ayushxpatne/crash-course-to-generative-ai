### Chapter 2: Numpy and Pandas Revision

#### 1. Introduction to Numpy
- **What is Numpy?**
  - Numpy is a powerful library for numerical computing in Python. It provides support for arrays, matrices, and many mathematical functions to operate on these data structures.

- **Creating Numpy Arrays**
  ```python
  import numpy as np

  # Creating arrays
  array1 = np.array([1, 2, 3])
  array2 = np.array([[1, 2, 3], [4, 5, 6]])

  print(array1)  # Output: [1 2 3]
  print(array2)  # Output: [[1 2 3]
                 #          [4 5 6]]
  ```

- **Basic Operations with Numpy**
  ```python
  # Array operations
  print(array1 + 2)  # Output: [3 4 5]
  print(array2 * 2)  # Output: [[ 2  4  6]
                     #          [ 8 10 12]]
  
  # Element-wise operations
  print(array1 + array1)  # Output: [2 4 6]
  print(array2 * array2)  # Output: [[ 1  4  9]
                          #          [16 25 36]]
  ```

- **Useful Numpy Functions**
  ```python
  # Arange, linspace, and reshape
  array3 = np.arange(10)
  array4 = np.linspace(0, 1, 5)
  array5 = array3.reshape(2, 5)

  print(array3)  # Output: [0 1 2 3 4 5 6 7 8 9]
  print(array4)  # Output: [0.   0.25 0.5  0.75 1.  ]
  print(array5)  # Output: [[0 1 2 3 4]
                 #          [5 6 7 8 9]]

  # Basic statistics
  print(np.mean(array3))  # Output: 4.5
  print(np.std(array3))   # Output: 2.8722813232690143
  print(np.sum(array3))   # Output: 45
  ```

#### 2. Introduction to Pandas
- **What is Pandas?**
  - Pandas is a powerful library for data manipulation and analysis. It provides data structures like Series and DataFrame which are useful for handling structured data.

- **Creating Pandas Series and DataFrames**
  ```python
  import pandas as pd

  # Creating a Series
  series = pd.Series([1, 2, 3, 4, 5])
  print(series)
  # Output:
  # 0    1
  # 1    2
  # 2    3
  # 3    4
  # 4    5
  # dtype: int64

  # Creating a DataFrame
  data = {
      "Name": ["Ayush", "Sarika", "Manjeet"],
      "Age": [21, 45, 50],
      "City": ["Pune", "Mumbai", "Delhi"]
  }
  df = pd.DataFrame(data)
  print(df)
  # Output:
  #      Name  Age    City
  # 0   Ayush   21    Pune
  # 1  Sarika   45  Mumbai
  # 2 Manjeet   50   Delhi
  ```

- **Basic Operations with DataFrames**
  ```python
  # Accessing data
  print(df["Name"])  # Output: Series with names
  print(df.iloc[0])  # Output: First row as Series

  # Adding a new column
  df["Country"] = ["India", "India", "India"]
  print(df)
  # Output:
  #      Name  Age    City Country
  # 0   Ayush   21    Pune   India
  # 1  Sarika   45  Mumbai   India
  # 2 Manjeet   50   Delhi   India

  # Filtering data
  filtered_df = df[df["Age"] > 30]
  print(filtered_df)
  # Output:
  #      Name  Age    City Country
  # 1  Sarika   45  Mumbai   India
  # 2 Manjeet   50   Delhi   India
  ```

- **Common DataFrame Operations**
  ```python
  # Reading from and writing to CSV
  df.to_csv("data.csv", index=False)
  new_df = pd.read_csv("data.csv")
  print(new_df)

  # Handling missing values
  df_with_nan = df.copy()
  df_with_nan.loc[1, "Age"] = None
  print(df_with_nan.isnull())  # Check for missing values

  df_filled = df_with_nan.fillna(df_with_nan.mean())
  print(df_filled)  # Fill missing values with mean
  ```

- **Group By and Aggregations**
  ```python
  # Grouping data
  grouped = df.groupby("City").mean()
  print(grouped)
  # Output:
  #         Age
  # City        
  # Delhi   50.0
  # Mumbai  45.0
  # Pune    21.0

  # Aggregating data
  aggregated = df.groupby("City").agg({"Age": ["mean", "max"]})
  print(aggregated)
  # Output:
  #           Age     
  #          mean max
  # City              
  # Delhi    50.0  50
  # Mumbai   45.0  45
  # Pune     21.0  21
  ```

### Some other important functions in Pandas that are frequently used for data manipulation and analysis:

#### 1. DataFrame Operations
- **Merging DataFrames**
  ```python
  df1 = pd.DataFrame({
      'A': ['A0', 'A1', 'A2'],
      'B': ['B0', 'B1', 'B2']
  })

  df2 = pd.DataFrame({
      'C': ['C0', 'C1', 'C2'],
      'D': ['D0', 'D1', 'D2']
  })

  merged_df = pd.concat([df1, df2], axis=1)
  print(merged_df)
  # Output:
  #     A   B   C   D
  # 0  A0  B0  C0  D0
  # 1  A1  B1  C1  D1
  # 2  A2  B2  C2  D2
  ```

- **Joining DataFrames**
  ```python
  left = pd.DataFrame({
      'key': ['K0', 'K1', 'K2', 'K3'],
      'A': ['A0', 'A1', 'A2', 'A3'],
      'B': ['B0', 'B1', 'B2', 'B3']
  })

  right = pd.DataFrame({
      'key': ['K0', 'K1', 'K2', 'K3'],
      'C': ['C0', 'C1', 'C2', 'C3'],
      'D': ['D0', 'D1', 'D2', 'D3']
  })

  joined_df = pd.merge(left, right, on='key')
  print(joined_df)
  # Output:
  #   key   A   B   C   D
  # 0  K0  A0  B0  C0  D0
  # 1  K1  A1  B1  C1  D1
  # 2  K2  A2  B2  C2  D2
  # 3  K3  A3  B3  C3  D3
  ```

#### 2. Advanced Data Manipulation
- **Pivot Tables**
  ```python
  df = pd.DataFrame({
      'A': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'],
      'B': ['one', 'one', 'two', 'two', 'one', 'one'],
      'C': ['small', 'large', 'small', 'small', 'small', 'large'],
      'D': [1, 2, 2, 3, 3, 4]
  })

  pivot_table = df.pivot_table(values='D', index=['A', 'B'], columns=['C'])
  print(pivot_table)
  # Output:
  # C          large  small
  # A   B                  
  # bar one     4.0    3.0
  #     two     NaN    3.0
  # foo one     2.0    1.0
  #     two     NaN    2.0
  ```

- **Stacking and Unstacking**
  ```python
  stacked = pivot_table.stack()
  print(stacked)
  # Output:
  # A    B   C    
  # bar  one large    4.0
  #          small    3.0
  #      two small    3.0
  # foo  one large    2.0
  #          small    1.0
  #      two small    2.0
  # dtype: float64

  unstacked = stacked.unstack()
  print(unstacked)
  # Output:
  # C          large  small
  # A   B                  
  # bar one     4.0    3.0
  #     two     NaN    3.0
  # foo one     2.0    1.0
  #     two     NaN    2.0
  ```

- **Melt**
  ```python
  df = pd.DataFrame({
      'A': {0: 'a', 1: 'b', 2: 'c'},
      'B': {0: 1, 1: 3, 2: 5},
      'C': {0: 2, 1: 4, 2: 6}
  })

  melted = pd.melt(df, id_vars=['A'], value_vars=['B', 'C'])
  print(melted)
  # Output:
  #    A variable  value
  # 0  a        B      1
  # 1  b        B      3
  # 2  c        B      5
  # 3  a        C      2
  # 4  b        C      4
  # 5  c        C      6
  ```

#### 3. Handling Missing Data
- **Drop Missing Data**
  ```python
  df_with_nan = pd.DataFrame({
      'A': [1, 2, None],
      'B': [None, 2, 3],
      'C': [1, None, None]
  })

  # Drop rows with any NaN values
  df_dropped = df_with_nan.dropna()
  print(df_dropped)
  # Output:
  # Empty DataFrame
  # Columns: [A, B, C]
  # Index: []

  # Drop columns with any NaN values
  df_dropped_col = df_with_nan.dropna(axis=1)
  print(df_dropped_col)
  # Output:
  #      C
  # 0  1.0
  ```

- **Fill Missing Data**
  ```python
  # Fill NaN values with a specific value
  df_filled = df_with_nan.fillna(0)
  print(df_filled)
  # Output:
  #      A    B    C
  # 0  1.0  0.0  1.0
  # 1  2.0  2.0  0.0
  # 2  0.0  3.0  0.0

  # Fill NaN values with mean of each column
  df_filled_mean = df_with_nan.fillna(df_with_nan.mean())
  print(df_filled_mean)
  # Output:
  #      A    B    C
  # 0  1.0  2.5  1.0
  # 1  2.0  2.0  1.0
  # 2  1.5  3.0  1.0
  ```

#### 4. Advanced Indexing
- **Setting and Resetting Index**
  ```python
  df = pd.DataFrame({
      'A': ['a', 'b', 'c'],
      'B': [1, 2, 3],
      'C': [4, 5, 6]
  })

  # Setting index
  df_indexed = df.set_index('A')
  print(df_indexed)
  # Output:
  #    B  C
  # A      
  # a  1  4
  # b  2  5
  # c  3  6

  # Resetting index
  df_reset = df_indexed.reset_index()
  print(df_reset)
  # Output:
  #    A  B  C
  # 0  a  1  4
  # 1  b  2  5
  # 2  c  3  6
  ```

- **MultiIndexing**
  ```python
  tuples = list(zip(*[['bar', 'bar', 'baz', 'baz'],
                      ['one', 'two', 'one', 'two']]))

  index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
  df_multi = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]}, index=index)
  print(df_multi)
  # Output:
  #               A  B
  # first second      
  # bar   one     1  5
  #       two     2  6
  # baz   one     3  7
  #       two     4  8

  # Accessing data in MultiIndex
  print(df_multi.loc['bar'])
  # Output:
  #        A  B
  # second      
  # one     1  5
  # two     2  6
  ```
