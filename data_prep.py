import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

def one_hot_encode_column(df, column_name):
    # Make a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Replace NaN values with '없음'
    df_copy[column_name].fillna('없음', inplace=True)

    # Split the specified column into lists
    df_copy[column_name] = df_copy[column_name].str.split(';')

    # Extract unique items for the MultiLabelBinarizer
    all_items = [item for sublist in df_copy[column_name] for item in sublist]
    unique_items = set(all_items)

    # Initialize the MultiLabelBinarizer with the unique items
    mlb = MultiLabelBinarizer(classes=list(unique_items))

    # One-hot encode the lists
    one_hot_encoded = mlb.fit_transform(df_copy[column_name])

    # Convert the result to a DataFrame
    df_one_hot = pd.DataFrame(one_hot_encoded, columns=mlb.classes_, index=df_copy.index)

    df_copy.drop(column_name,axis = 1,inplace = True)

    # Join the one-hot encoded DataFrame back to the copied DataFrame
    df_copy = pd.concat([df_copy, df_one_hot], axis=1)

    return df_copy

def one_hot_encode_numeric_range(df, column_name, range_size):
    # Make a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Bin the numeric values of the specified column into ranges
    bins = range(0, int(df_copy[column_name].max()) + range_size, range_size)
    labels = [f"{i}-{i+range_size-1}" for i in bins[:-1]]
    df_copy[f"{column_name}_binned"] = pd.cut(df_copy[column_name], bins=bins, labels=labels, right=False)

    # One-hot encode the binned column
    df_one_hot = pd.get_dummies(df_copy[f"{column_name}_binned"], prefix=column_name)

    # Join the one-hot encoded DataFrame back to the copied DataFrame
    df_copy = pd.concat([df_copy, df_one_hot], axis=1)

    df_copy.drop(column_name,axis = 1,inplace = True)
    # Drop the temporary binned column
    df_copy.drop(f"{column_name}_binned", axis=1, inplace=True)

    return df_copy

def one_hot_encode_date(df, column_name):
    # Make a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Convert the column to datetime format (if it's not already)
    # Coerce errors to NaT
    df_copy[column_name] = pd.to_datetime(df_copy[column_name], errors='coerce')

    # Handle NaT values (you can replace them with a default date or drop them)
    # Here, I'm replacing them with a default date '1900-01-01'
    df_copy[column_name].fillna(pd.Timestamp('1900-01-01'), inplace=True)

    # Extract year-month from the date and store it in a new column
    df_copy[f"{column_name}_ym"] = df_copy[column_name].dt.strftime('%Y-%m')

    # One-hot encode the year-month column
    df_one_hot = pd.get_dummies(df_copy[f"{column_name}_ym"], prefix=column_name)

    # Join the one-hot encoded DataFrame back to the copied DataFrame
    df_copy = pd.concat([df_copy, df_one_hot], axis=1)

    df_copy.drop(column_name,axis = 1,inplace = True)

    # Drop the temporary year-month column
    df_copy.drop(f"{column_name}_ym", axis=1, inplace=True)

    return df_copy

def pick_n_ones(row, n):
    # Get indices of ones
    ones_indices = np.where(row == 1)[0]

    # If there are fewer than n ones, return the row as is
    if len(ones_indices) <= n:
        return row

    # Randomly select n indices to keep as ones
    keep_indices = np.random.choice(ones_indices, size=n, replace=False)

    # Set all other ones to zero
    for idx in ones_indices:
        if idx not in keep_indices:
            row[idx] = 0

    return row