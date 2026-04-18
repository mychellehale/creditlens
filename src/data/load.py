import pandas as pd
import kagglehub
from pathlib import Path


def get_kaggle_data(dataset: str) -> pd.DataFrame:
    '''
    Downloads a dataset from Kaggle and returns it as a pandas DataFrame.
    Locates the first CSV file in the downloaded directory automatically.

    :param dataset: Kaggle dataset identifier in the format 'owner/dataset-name'
    :type dataset: str
    :return: DataFrame containing the downloaded dataset
    :rtype: pd.DataFrame
    '''
    path = kagglehub.dataset_download(dataset)
    csv_file = next(Path(path).glob("*.csv"))
    df = pd.read_csv(csv_file)
    return df