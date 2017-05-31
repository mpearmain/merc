import pandas as pd

def remove_constant_cols(df):
    """
    :param df: Pandas dataFrame to remove constant cols for
    :return: Pandas dataframe with constant cols removed
    """
    df = df.loc[:, (df != df.ix[0]).any()]
    return df

