import pandas as pd

def load_interaction_data(file_path):
    """
    Load user-item interaction data from a CSV file.
    :param file_path: Path to the CSV file.
    :return: DataFrame containing interaction data.
    """
    return pd.read_csv(file_path)
