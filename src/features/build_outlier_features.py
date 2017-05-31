

def recode_outlier(data, col, ulimit):
    data[col].loc[data[col] > ulimit] = ulimit
    return data[col].values
