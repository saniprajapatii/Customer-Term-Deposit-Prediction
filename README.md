# Term-Deposit-Prediction-


**1.We have used multiple libraries to perform EDA**

1. **Import NumPy:-** Import the NumPy library for numerical operations and array handling.

2. **Import Pandas:-** Import the Pandas library for data manipulation and analysis using data frames.

3. **Import Seaborn:-** Import the Seaborn library for statistical data visualization.

4. **Import Matplotlib:-** Import the Matplotlib library for creating static, interactive, and animated visualizations in Python.

5.** Enable Inline Plotting:-** This IPython magic command enables inline plotting in Jupyter notebooks, allowing Matplotlib plots to be displayed directly below the code cells.

6.** Suppress Warnings:-** This code suppresses warnings to prevent them from being displayed during the execution of the code, ensuring a cleaner output in the notebook or console.

**2. Data Profiling**

1. **Read Training Data from CSV:-** Read the training data from the CSV file named "train.csv" into the Pandas DataFrame train_df.

2.**Read Testing Data from CSV:-** Read the testing data from the CSV file named "test.csv" into the Pandas DataFrame test_df.

3. **Display First Few Rows of Training Data:-** Display the first few rows of the training data using the head() function. This allows you to inspect the structure and contents of the training dataset, ensuring successful loading and understanding the format of the data.

4. **Get DataFrame Shape:-** This code retrieves the shape of the DataFrame train_df. It returns a tuple where the first element represents the number of rows and the second element represents the number of columns in the DataFrame. This information is valuable for understanding the dataset's size and structure.

5. **Display DataFrame Summary:-** This code calls the info() method on the DataFrame train_df. It provides a summary of the DataFrame, including the data types of each column, the number of non-null values, and memory usage. This information is helpful for understanding the data types of features, detecting missing values, and assessing the overall data quality.

4. **Generate Summary Statistics:-** This code calls the describe() method on the DataFrame train_df with the include='all' parameter. It generates summary statistics for all columns, including numerical and categorical data. For numerical columns, it provides count, mean, standard deviation, minimum, 25th percentile (Q1), median (50th percentile or Q2), 75th percentile (Q3), and maximum values. For categorical columns, it includes count, unique values, top (most frequent value), and frequency of the top value. This information is valuable for understanding the distribution and central tendencies of the data.

5. **Check for Null Values:-** This code evaluates each element in train_df and returns a Series indicating the count of null values in each column. It helps identify columns with missing data, which is essential for data preprocessing and handling missing values appropriately in the analysis or modeling process.

**3.Analysis**
1. **Create Count Plot:-** This code uses Seaborn's countplot function to create a bar plot. The x parameter specifies the variable to be plotted on the x-axis the data parameter specifies the DataFrame (train_df in this case), and the hue parameter specifies the variable to differentiate the bars based on ("y" in this case).

2. **Create Pie Chart:-** This code creates a pie chart using Matplotlib's plt.pie function. The first argument train_df.marital.value_counts() calculates the counts of unique values in the 'marital' column. The labels parameter specifies the labels for each category. The autopct parameter formats the percentage labels on the chart.

3. **Calculate Correlation Matrix:-** This code computes the correlation matrix for the train_df DataFrame using the corr() method. The correlation matrix represents the relationships between numerical variables in the dataset.

4. There is no multicollinearity between independent variables

**4.Feature Encoding**

1.**Concatenate Data Frames:-** Concatenate train_df and test_df data frames along rows (axis=0). The ignore_index=True parameter resets the index of the resulting data frame, ensuring a continuous index without considering the source data frame indices.

2. **Apply One-Hot Encoding:-** Use pd.get_dummies to convert categorical columns specified in the columns parameter into binary columns (dummy variables). The drop_first=True parameter drops the first category in each original categorical variable, avoiding multicollinearity in some machine learning models.

3.**Display Modified DataFrame:-** Display the first few rows of the modified DataFrame df after one-hot encoding. This allows you to inspect the changes made to the categorical columns and confirm the successful transformation.


1. **Assign Target Variable:-** Create a variable target containing the 'y' column, which represents the target variable you want to predict.

2. **Remove Target Variable from DataFrame:-** Remove the 'y' column from the DataFrame df along the columns (axis=1), ensuring that the DataFrame no longer contains the target variable. This modified DataFrame can now be used for feature selection and modeling purposes.

**5.Feature Scaling**


