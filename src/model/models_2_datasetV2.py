import fastparquet
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from gestalt.estimator_wrappers.wrap_xgb import XGBRegressor
from gestalt.stackers.stacking import GeneralisedStacking
from keras.layers import Dense, Dropout, BatchNormalization, Activation
# keras
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.ensemble import RandomForestRegressor
# model evaluation
from sklearn.metrics import r2_score

# supportive models
# feature selection (from supportive model)

# to make results reproducible
seed = 42 # was 42

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

# Fix the folds - generates skf object
from sklearn.model_selection import KFold

skf = KFold(shuffle=True, n_splits=15, random_state=260681)

"""
Setting up keras model.
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

# define custom R2 metrics for Keras backend
from keras import backend as K

# prepare callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,  # was 10
        verbose=1),

    ModelCheckpoint(
        model_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=0)
]

def r2_keras(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


# base model architecture definition
def model():
    model = Sequential()
    # input layer
    model.add(Dense(input_dims, input_dim=input_dims))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    # hidden layers
    model.add(Dense(input_dims))
    model.add(BatchNormalization())
    model.add(Activation(act_func))
    model.add(Dropout(0.4))

    model.add(Dense(input_dims // 2))
    model.add(BatchNormalization())
    model.add(Activation(act_func))
    model.add(Dropout(0.4))

    model.add(Dense(input_dims // 4, activation=act_func))

    # output layer (y_pred)
    model.add(Dense(1, activation='linear'))

    # compile this model
    model.compile(loss='mean_squared_error',  # one may use 'mean_absolute_error' as alternative
                  optimizer='adam',
                  metrics=[r2_keras]  # you can add several if needed
                  )

    # Visualize NN architecture
    print(model.summary())
    return model


# initialize input dimension
input_dims = train.shape[1] - 1

# activation functions for hidden layers
act_func = 'tanh'  # could be 'relu', 'sigmoid', ...tanh

# make np.seed fixed
np.random.seed(seed)

# initialize estimator, wrap model in KerasRegressor
estimators = {KerasRegressor(build_fn=model, nb_epoch=300, batch_size=20, verbose=1): 'MLP1' + BUILD_NAME}
merc = GeneralisedStacking(base_estimators_dict=estimators,
                           estimator_type='regression',
                           feval=r2_score,
                           stack_type='s',
                           folds_strategy=skf)
merc.fit(train, y_train)
lvl1meta_train_regressor = merc.meta_train
lvl1meta_test_regressor = merc.predict(test)

print('Writing Parquets')
# store
fastparquet.write('./data/processed/metalvl1/xtrain_metalvl1_models2_' + BUILD_NAME + '.parq', lvl1meta_train_regressor,
                  write_index=False)
fastparquet.write('./data/processed/metalvl1/xtest_metalvl1_models2_' + BUILD_NAME + '.parq', lvl1meta_test_regressor,
                  write_index=False)
print('Finished')
