import fastparquet
import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor

from gestalt.stackers.stacking import GeneralisedStacking
from gestalt.estimator_wrappers.wrap_xgb import XGBRegressor
from sklearn.metrics import r2_score

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

# Fix the folds - generates skf object
from sklearn.model_selection import KFold
skf = KFold(shuffle=True, n_splits=15, random_state=260681)

"""
build_datasetV2 can use all regression 
"""

# Read the base data
print('Loading data ...')
BUILD_NAME = '_build_datasetV2'
train = fastparquet.ParquetFile('./data/processed/xtrain' + BUILD_NAME + '.parq').to_pandas()
test = fastparquet.ParquetFile('./data/processed/xtest' + BUILD_NAME + '.parq').to_pandas()
print('Loaded')

y_train = train['y'].values
y_mean = np.mean(y_train)
id_train = train['ID'].values
id_test = test['ID'].values
train = train.drop(['ID', 'y'], axis=1)
test = test.drop(['ID'], axis=1)

print("Ready to model")


def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)

params1 = {"eta": 0.05, "subsample": 0.9, "min_child_weight": 1, "colsample_bytree": 0.8,
           "max_depth": 8, 'silent': 1, "n_jobs": 12}
params2 = {"eta": 0.025, "subsample": 0.95, "min_child_weight": 3, "colsample_bytree": 0.7,
           "max_depth": 10, 'silent': 1, "n_jobs": 12}
params3 = {"eta": 0.01, "subsample": 0.9, "min_child_weight": 2, "colsample_bytree": 0.9, "gamma":0.01,
           "max_depth": 4, 'silent': 1, "n_jobs": 12}
params4 = {'eta': 0.025, 'max_depth': 2, 'subsample': 0.95, 'colsample_bytree': 0.8, 'objective': 'reg:linear',
           'eval_metric': 'rmse', 'silent': 1, "n_jobs": 12, "booster": "dart"}
params5 = {'n_trees': 500, 'eta': 0.005, 'max_depth': 4, 'subsample': 0.95, 'objective': 'reg:linear',
              'eval_metric': 'rmse', 'base_score': y_mean, 'silent': 1, "n_jobs": -1}


estimators = {RandomForestRegressor(n_estimators=100, n_jobs=12, random_state=42, max_features=0.8): 'RFR1' + BUILD_NAME,
              XGBRegressor(num_round=5000, verbose_eval=False, params=params1, early_stopping_rounds=50,
                           eval_metric=xgb_r2_score): 'XGB1' + BUILD_NAME,
              XGBRegressor(num_round=5000, verbose_eval=False, params=params2, early_stopping_rounds=50,
                           eval_metric=xgb_r2_score): 'XGB2' + BUILD_NAME,
              XGBRegressor(num_round=5000, verbose_eval=False, params=params3, early_stopping_rounds=50,
                           eval_metric=xgb_r2_score): 'XGB3' + BUILD_NAME,
              XGBRegressor(num_round=5000, verbose_eval=False, params=params4, early_stopping_rounds=50): 'XGB4' + BUILD_NAME,
              XGBRegressor(num_round=700, verbose_eval=False, params=params5): 'XGB5' + BUILD_NAME,
              ElasticNet(alpha=0.01, l1_ratio=0.1,): 'ElasticNet1' + BUILD_NAME,
              ElasticNet(alpha=0.1, l1_ratio=0.01, ): 'ElasticNet2' + BUILD_NAME}

merc = GeneralisedStacking(base_estimators_dict=estimators,
                           estimator_type='regression',
                           feval=r2_score,
                           stack_type='s',
                           folds_strategy=skf)
merc.fit(train, y_train)

lvl1meta_train_regressor = merc.meta_train
lvl1meta_test_regressor = merc.predict(test)

lvl1meta_train_regressor['ID'] = id_train
lvl1meta_train_regressor['y'] = y_train
lvl1meta_test_regressor['ID'] = id_test


print('Writing Parquets')
# store
fastparquet.write('./data/processed/metalvl1/xtrain_metalvl1' + BUILD_NAME + '.parq', lvl1meta_train_regressor,
                  write_index=False)
fastparquet.write('./data/processed/metalvl1/xtest_metalvl1' + BUILD_NAME + '.parq', lvl1meta_test_regressor,
                  write_index=False)
print('Finished')