import fastparquet
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from gestalt.stackers.stacking import GeneralisedStacking
from gestalt.estimator_wrappers.wrap_xgb import XGBRegressor
from sklearn.metrics import r2_score

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

# Fix the folds - generates skf object
from sklearn.model_selection import KFold
skf = KFold(shuffle=True, n_splits=20, random_state=260681)

"""
build_datasetV1 has had no treatment to the object or count columns and so is unsuitable for any models other than 
tree based models, or hash based models (i.e FTRLProximal or FMs)
"""

# Read the base data
print('Loading data ...')
BUILD_NAME = '_build_datasetV1'
train = fastparquet.ParquetFile('./data/processed/xtrain' + BUILD_NAME + '.parq').to_pandas()
test = fastparquet.ParquetFile('./data/processed/xtest' + BUILD_NAME + '.parq').to_pandas()
print('Loaded')

y_train = train['y'].values
id_train = train['ID'].values
id_test = test['ID'].values
train = train.drop(['ID', 'y'], axis=1)
test = test.drop(['ID'], axis=1)

print("Ready to model")


def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)

params1 = {"eta": 0.05, "subsample": 0.8, "min_child_weight": 1, "colsample_bytree": 0.8,
           "max_depth": 8, 'silent': 1, "n_jobs": 12}
params2 = {"eta": 0.025, "subsample": 0.7, "min_child_weight": 3, "colsample_bytree": 0.7,
           "max_depth": 10, 'silent': 1, "n_jobs": 12}
params3 = {"eta": 0.01, "subsample": 0.9, "min_child_weight": 2, "colsample_bytree": 0.9, "gamma":0.5,
           "max_depth": 4, 'silent': 1, "n_jobs": 12}


estimators = {RandomForestRegressor(n_estimators=100, n_jobs=12, random_state=42, max_features=0.8): 'RFR1' + BUILD_NAME,
              XGBRegressor(num_round=5000, verbose_eval=False, params=params1, early_stopping_rounds=25,
                           eval_metric=xgb_r2_score): 'XGB1' + BUILD_NAME,
              XGBRegressor(num_round=5000, verbose_eval=False, params=params2, early_stopping_rounds=25,
                           eval_metric=xgb_r2_score): 'XGB2' + BUILD_NAME,
              XGBRegressor(num_round=5000, verbose_eval=False, params=params3, early_stopping_rounds=50,
                           eval_metric=xgb_r2_score): 'XGB3' + BUILD_NAME}

merc = GeneralisedStacking(base_estimators_dict=estimators,
                           estimator_type='regression',
                           feval=r2_score,
                           stack_type='s',
                           folds_strategy=skf)
merc.fit(train, y_train)

lvl1meta_train_regressor = merc.meta_train
lvl1meta_test_regressor = merc.predict(test)

###################### Level 2 Stacking #################
clf = {LinearRegression(n_jobs=-1, ): 'Linear_regression_lvl2' + BUILD_NAME}
merc2 = GeneralisedStacking(base_estimators_dict=clf,
                             estimator_type='regression',
                             feval=r2_score,
                             stack_type='s',
                             folds_strategy=skf)
merc2.fit(lvl1meta_train_regressor, y_train)
lvl2meta_test = merc2.predict(lvl1meta_test_regressor)


sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = lvl1meta_test_regressor['XGB1_build_datasetV1']
sub.to_csv('./data/output/regression_stacking_test.csv', index=False)
