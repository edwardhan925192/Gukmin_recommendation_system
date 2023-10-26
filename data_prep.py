import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

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

def extract_unique_words(data, column_name):
    unique_words = set()

    data[column_name].fillna('none', inplace=True)

    for row in data[column_name]:
        words = row.split(';')
        unique_words.update(words)

    return unique_words

def map_dict(input_list1, input_list2):
    # Ensure the two lists have the same length
    if len(input_list1) != len(input_list2):
        raise ValueError("The two lists must have the same length.")

    # Create the dictionary using a dictionary comprehension
    result_dict = {input_list1[i]: input_list2[i] for i in range(len(input_list1))}

    return result_dict

def translate(df,column, translation_dict):
    replaced_values = []
    for entry in df[column]:
        # Split the entry by " ; "
        words = entry.split(";")
        # Replace each word with its translation from the dictionary
        translated_words = [translation_dict.get(word, word) for word in words]
        # Join the translated words back with " ; "
        replaced_values.append(";".join(translated_words))

    df[column] = replaced_values
    return df

def map_cluster(df, column, mapping_dict):
    mapped_values = []
    for entry in df[column]:
        # Split the entry by " ; "
        words = entry.split(";")
        # Map each word to its corresponding number from the dictionary
        numbers = [str(mapping_dict.get(word, word)) for word in words]
        # Join the numbers back with " ; "
        mapped_values.append(";".join(numbers))

    df[column] = mapped_values
    return df
    
def onehot_full(df, column_name):
    # One-hot encode the specified column
    one_hot = pd.get_dummies(df[column_name], prefix=column_name)

    # Drop the original column from the DataFrame
    df = df.drop(column_name, axis=1)

    # Concatenate the one-hot encoded columns to the original DataFrame
    df = pd.concat([df, one_hot], axis=1)

    return df

import numpy as np
import torch.nn.functional as F

# ==================== Input should be in numpy format ===================== #
def train_prep(matrix, n):
    matrix_copy = matrix.copy()
    for row in matrix_copy:
        ones_indices = np.where(row == 1)[0]

        if len(ones_indices) <= n:
            continue

        zero_indices = np.random.choice(ones_indices, size=n, replace=False)

        for idx in zero_indices:
            row[idx] = 0

    return matrix_copy

def find_ones(matrix):
    result = []
    for row in matrix:
        indices = np.where(row == 1)[0]
        if indices.size:
            result.append(tuple(indices))
        else:
            result.append('None')
    return result

def get_matrix(matrix, indexes_list):

    if isinstance(matrix, np.ndarray):
        matrix = torch.tensor(matrix)
    # Ensure that the number of rows in the matrix matches the length of indexes_list
    assert matrix.shape[0] == len(indexes_list), "Number of rows in matrix must match length of indexes_list"

    # Process each row in the matrix
    for i in range(matrix.shape[0]):
        # Calculate the number of indexes for the current row
        num_indexes = len(indexes_list[i])

        # Get the indices of the top values for the current row
        _, top_indices = torch.topk(matrix[i], num_indexes)

        # Create a mask with 0s everywhere
        mask = torch.zeros_like(matrix[i])

        # Set the top indices in the mask to 1
        mask[top_indices] = 1

        # Update the row using the mask
        matrix[i] = mask

    return matrix

def bce_loss(output, target):

    # Compute the binary cross-entropy loss without reduction
    loss = F.binary_cross_entropy_with_logits(output, target, reduction='none')

    # Create a mask where the target matrix has 1s
    mask = target == 1

    # Apply the mask to the loss
    masked_loss = loss * mask.float()

    # Compute and return the final loss value
    return masked_loss.sum() / mask.sum()

# TQDM version 
from tqdm import tqdm

def matching_rows(data, column_lists, return_column):
    results = []
    
    # Wrap the range with tqdm to show progress
    for i in tqdm(range(len(data)), desc="Processing rows"):
        current_matches = []
        
        # For the first set of columns, we just need exact matches
        mask1 = (data[column_lists[0]] == data.iloc[i][column_lists[0]].values).all(axis=1)
        
        # For the second set of columns, we need to check if values are within a range
        mask2 = ((data[column_lists[1]] - data.iloc[i][column_lists[1]].values).abs() <= 5).all(axis=1)
        
        # For the remaining sets of columns, we just need exact matches
        other_masks = [(data[cols] == data.iloc[i][cols].values).all(axis=1) for cols in column_lists[2:]]
        
        # Combine all masks
        final_mask = mask1 & mask2 & all(other_masks)
        
        # Exclude the current row
        final_mask.iloc[i] = False
        
        # Append the matched rows' return_column values to current_matches
        current_matches.extend(data.loc[final_mask, return_column].tolist())
        
        results.append(current_matches)

    return results
