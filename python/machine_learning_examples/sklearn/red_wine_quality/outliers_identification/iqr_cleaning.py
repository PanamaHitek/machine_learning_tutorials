import pandas as pd

# Load the dataset
df = pd.read_csv('../../../../../datasets/red_wine_quality/dataset.csv')


def get_outliers_using_iqr(data):
    """
    Identify and return outliers based on IQR for each column in the dataframe.

    :param data: DataFrame
    :return: DataFrame with outliers
    """
    outliers = pd.DataFrame(columns=data.columns)

    for column in data:
        if column == "quality":  # Skip 'quality' column
            continue

        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers_for_column = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        outliers = pd.concat([outliers, outliers_for_column])

    return outliers.drop_duplicates()


def get_farthest_outlier(data, outliers_df):
    """
    Identify the row index of the farthest outlier in the dataframe.

    :param data: DataFrame
    :param outliers_df: DataFrame containing outliers
    :return: Index of the farthest outlier
    """
    distances = []
    for column in data.columns:
        if column == "quality":  # Skip 'quality' column
            continue

        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outlier_distances = outliers_df[column].apply(lambda x: min(abs(x - lower_bound), abs(x - upper_bound)))
        distances.append(outlier_distances)

    distances_df = pd.concat(distances, axis=1)
    max_distance_idx = distances_df.sum(axis=1).idxmax()

    return max_distance_idx


# Identify and remove outliers
while True:
    outliers = get_outliers_using_iqr(df)
    if len(outliers) == 0:
        break

    outlier_to_remove = get_farthest_outlier(df, outliers)
    df = df.drop(outlier_to_remove)

# Save the cleaned dataset
df.to_csv('../../../../../datasets/red_wine_quality/clean_dataset_iqr.csv', index=False)
