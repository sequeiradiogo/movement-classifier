# -*- coding: utf-8 -*-
"""
Feature Filtering Script
------------------------
Extracts data from a given directory using the `extract` function,
filters specific columns, and saves the result as a new CSV file.

Created: Dec 17, 2024
Author: Diogo Sequeira
"""

from extract_files import extract
import pandas as pd


def filter_features(path: str) -> pd.DataFrame:
    """
    Extracts features from a given directory and filters specific columns.

    Parameters
    ----------
    path : str
        Path to the directory containing the input files.

    Returns
    -------
    pd.DataFrame
        The filtered DataFrame containing only selected columns.
    """
    output_path = 'filtered_features.csv'

    # Extract data using custom extraction function
    df = extract(path)

    # Columns to keep (selected using Orange 3)
    selected_columns = [
        'File',
        'Gyro_Z_Area under the curve',
        'Gyro_X_Slope',
        'Gyro_X_Sum absolute diff',
        'Gyro_X_Signal distance',
        'Accel_Z_Mean',
        'Class'
    ]

    # Check which columns exist
    available_columns = [col for col in selected_columns if col in df.columns]
    missing = set(selected_columns) - set(available_columns)
    if missing:
        print(f"Warning: Missing columns in input data: {', '.join(missing)}")

    # Filter the DataFrame
    filtered_df = df[available_columns]

    # Save result
    filtered_df.to_csv(output_path, index=False, sep=';')
    print(f"Filtered file saved to: {output_path}")

    return filtered_df


if __name__ == "__main__":
    # Example usage (replace with your actual path)
    directory = r"C:\Users\35193\OneDrive\Ambiente de Trabalho\Universidade\5º ano 1º semestre\AAI\dados_novos"
    filter_features(directory)
