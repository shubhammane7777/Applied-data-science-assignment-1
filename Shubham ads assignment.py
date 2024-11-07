import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew, kurtosis

# Load dataset
data_path = '/Users/abc/Documents/Assignment dataset/world_bank_dataset.csv'
df = pd.read_csv(data_path)

# Convert columns to numeric where possible, forcing errors to NaN
df = df.apply(pd.to_numeric, errors='coerce')

# Print the columns to verify correct names
print("Columns in the dataset:", df.columns)


# Function for generating descriptive statistics
def calculate_statistics(df):
    # General statistics
    stats = df.describe()
    # Additional statistics
    skewness = df.apply(lambda x: skew(x.dropna()), axis=0)
    kurt = df.apply(lambda x: kurtosis(x.dropna()), axis=0)
    
    # Concatenate statistics (use pd.concat instead of append)
    summary = pd.concat([stats, skewness.rename('skewness'), kurt.rename('kurtosis')], axis=0)
    return summary

# Function to plot a histogram
def plot_histogram(df, column):
    plt.figure(figsize=(8, 6))
    plt.hist(df[column].dropna(), bins=20, color='red', edgecolor='black')
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

# Function to plot a line chart
def plot_line_chart(df, x_column, y_column):
    plt.figure(figsize=(10, 6))
    plt.plot(df[x_column].dropna(), df[y_column].dropna(), marker='o', color='orange')
    plt.title(f'Line Chart of {y_column} over {x_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.show()

# Function to plot a heatmap (correlation matrix) using matplotlib only
def plot_heatmap(df):
    plt.figure(figsize=(10, 8))
    corr_matrix = df.corr()
    plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    
    # Adding labels for heatmap
    plt.xticks(ticks=np.arange(len(corr_matrix.columns)), labels=corr_matrix.columns, rotation=90)
    plt.yticks(ticks=np.arange(len(corr_matrix.columns)), labels=corr_matrix.columns)
    plt.title('Correlation Matrix Heatmap')
    plt.show()

# Generate statistics
stats_summary = calculate_statistics(df)
print("Statistical Summary:\n", stats_summary)

# Plot histogram for a selected column
plot_histogram(df, column='Population') 

# Plot line chart for selected x and y columns
plot_line_chart(df, x_column='Year', y_column='Population')  

# Plot heatmap
plot_heatmap(df)

