import pandas as pd
from pathlib import Path

def load_data(file_path: Path) -> pd.DataFrame:
    """Loads election data from a CSV file."""
    try:
        # Added low_memory=False to handle DtypeWarning from your notebook
        df = pd.read_csv(file_path, low_memory=False)
        print(f"Data loaded successfully from {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return pd.DataFrame()

def filter_years(df: pd.DataFrame, years: list) -> pd.DataFrame:
    """Filters the DataFrame to include only specified election years."""
    print(f"Filtering data for years: {years}")
    return df[df['Year'].isin(years)].copy()

def handle_missing_values(df: pd.DataFrame, subset_cols: list) -> pd.DataFrame:
    """Drops rows with missing values in specified columns."""
    print(f"Dropping rows with missing values in: {subset_cols}")
    df.dropna(subset=subset_cols, inplace=True)
    return df

def standardize_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Converts specified columns to lowercase."""
    print(f"Standardizing columns: {cols}")
    for col in cols:
        df[col] = df[col].str.lower()
    return df

def select_and_rename_features(df: pd.DataFrame, columns_map: dict) -> pd.DataFrame:
    """Selects and renames columns for the final dataset."""
    print("Selecting and renaming final features.")
    df_selected = df[list(columns_map.keys())].copy()
    df_selected.rename(columns=columns_map, inplace=True)
    return df_selected

def calculate_vote_share(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates and adds the Vote_Share column."""
    print("Calculating Vote Share.")
    # Ensure Valid_Votes is not zero to avoid division errors
    df['Vote_Share'] = df.apply(
        lambda row: (row['Votes'] / row['Valid_Votes']) * 100 if row['Valid_Votes'] > 0 else 0,
        axis=1
    )
    return df