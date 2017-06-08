import fastparquet
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from gestalt.stackers.stacking import GeneralisedStacking
from gestalt.estimator_wrappers.wrap_xgb import XGBRegressor
from sklearn.metrics import r2_score

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

# Fix the folds - generates skf object
from sklearn.model_selection import KFold
skf = KFold(shuffle=True, n_splits=15, random_state=260681)

# Read the base data
print('Loading data ...')
# Read all the data from the dir and concat

BUILD_NAME = '_build_dataset_metalevel2V1'
BASE_COLS = ["y", "ID", "X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"]

train = fastparquet.ParquetFile('./data/processed/metalvl2/xtrain' + BUILD_NAME + '.parq').to_pandas()
test = fastparquet.ParquetFile('./data/processed/metalvl2/xtest' + BUILD_NAME + '.parq').to_pandas()
print('Loaded')

y_train = train['y'].values
y_mean = np.mean(y_train)
id_train = train['ID'].values
id_test = test['ID'].values

train = train.drop(["y", "X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"], axis=1)
test = test.drop(["X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"], axis=1)


# train = train.drop(['ID', 'y'], axis=1)
# test = test.drop(['ID'], axis=1)

print("Ready to model")


def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)


params1 = {'eta': 0.005, 'max_depth': 4, 'subsample': 0.95, 'objective': 'reg:linear',
              'eval_metric': 'rmse', 'base_score': y_mean, 'silent': 1, "n_jobs": -1}

estimators = {XGBRegressor(num_round=5000, verbose_eval=False, params=params1, early_stopping_rounds=25,
                           eval_metric=xgb_r2_score): 'XGB1' + BUILD_NAME,
              ElasticNet(alpha=0.01, l1_ratio=0.05): 'ElasticNet1' + BUILD_NAME,
              ElasticNet(alpha=0.1, l1_ratio=0.5): 'ElasticNet2' + BUILD_NAME,
              ElasticNet(alpha=0.5, l1_ratio=0.01): 'ElasticNet3' + BUILD_NAME}

merc = GeneralisedStacking(base_estimators_dict=estimators,
                           estimator_type='regression',
                           feval=r2_score,
                           stack_type='s',
                           folds_strategy=skf)
merc.fit(train, y_train)
lvl1meta_train_regressor = merc.meta_train
lvl1meta_test_regressor = merc.predict(test)

###################### Level 2 Stacking #################
clf = {LinearRegression(n_jobs=-1): 'Linear_regression_lvl2' + BUILD_NAME}
merc2 = GeneralisedStacking(base_estimators_dict=clf,
                            estimator_type='regression',
                            feval=r2_score,
                            stack_type='cv',
                            folds_strategy=skf)

merc2 = LinearRegression(n_jobs=-1)
merc2.fit(lvl1meta_train_regressor, y_train)
lvl2meta_test = merc2.predict(lvl1meta_test_regressor)

sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = lvl2meta_test
sub.to_csv('./data/output/megaID_test.csv', index=False)
