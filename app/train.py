# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import sklearn
import joblib

from sklearn.preprocessing import TargetEncoder
import xgboost as xgb

print(f'pandas=={pd.__version__}')
print(f'numpy=={np.__version__}')
print(f'sklearn=={sklearn.__version__}')

def load_data():
  df_full_train = pd.read_csv('data/train.csv')
  df_macro = pd.read_csv('data/macro.csv')

  df_full_train.columns = df_full_train.columns.str.lower().str.replace(' ','_')

  for col in ['full_sq', 'floor', 'max_floor', 'material', 'num_room']:
      df_full_train[col] = df_full_train[col].astype('float64')

  categorical_col = list(df_full_train.dtypes[df_full_train.dtypes == 'object'].index)

  for col in categorical_col:
    df_full_train[col] = df_full_train[col].str.lower().str.replace(' ','_')

  df_full_train['timestamp'] = pd.to_datetime(df_full_train['timestamp'])

  df_full_train = df_full_train[df_full_train.price_doc > 1000000]

  df_full_train['log_price_doc'] = np.log1p(df_full_train.price_doc)

  df_macro.columns = df_macro.columns.str.lower().str.replace(' ','_')

  df_macro['timestamp'] = pd.to_datetime(df_macro['timestamp'])
  df_macro = df_macro[(df_macro['timestamp'] >= '2011-08-20') & (df_macro['timestamp'] <= '2016-05-30')]

  macro_selected_col = ['cpi', 'usdrub']
  df_macro = df_macro[ ['timestamp'] + macro_selected_col]

  df_full_train = df_full_train.merge(df_macro, on='timestamp', how='left')

  categorical_col = ['sub_area',
                     'ecology',
                     'product_type']

  numerical_col = ['full_sq',
                   'kremlin_km',
                   'metro_km_walk',
                   'school_km',
                   'kindergarten_km',
                   'railroad_station_avto_min',
                   'usdrub',
                   'cpi']

  target = ['log_price_doc']

  df_full_train = df_full_train[categorical_col + numerical_col + target] 

  df_full_train['metro_km_walk'] = df_full_train['metro_km_walk'].fillna(df_full_train['metro_km_walk'].median())


  return df_full_train

def train_model(df_full_train):

  categorical_col = ['sub_area',
                     'ecology',
                     'product_type']

  numerical_col = ['full_sq',
                   'kremlin_km',
                   'metro_km_walk',
                   'school_km',
                   'kindergarten_km',
                   'railroad_station_avto_min',
                   'usdrub',
                   'cpi']

  encoder = TargetEncoder(
    target_type='continuous',
    smooth='auto')

  full_train_encoded_cats = encoder.fit_transform(
    df_full_train[categorical_col],
    df_full_train['log_price_doc'])

  df_full_train_encoded = pd.DataFrame(full_train_encoded_cats)

  new_column_names = []
  for col in categorical_col:
      new_column_names.append(f'{col}_encoded')

  df_full_train_encoded.columns = new_column_names

  df_full_train_encoded[numerical_col] = df_full_train[numerical_col]

  df_full_train_encoded['log_price_doc'] = df_full_train['log_price_doc']

  feature_names = df_full_train_encoded.drop('log_price_doc', axis = 1).columns.to_list()       

  y_full_train = df_full_train_encoded['log_price_doc'].values
  X_full_train = df_full_train_encoded.drop('log_price_doc', axis = 1).values     

  dfulltrain = xgb.DMatrix(X_full_train, label=y_full_train,
                           feature_names = feature_names)

  xgb_params = {
      'eta' : 0.05,
      'max_depth': 4,
      'min_child_weight' : 10,

      'objective': 'reg:squarederror',
      'nthread': -1,

      'seed': 42,
      'verbosity': 1,

      'eval_metric': 'rmse',

  }

  model = xgb.train(xgb_params,dfulltrain,
                    num_boost_round = 465)



  return {
        'categorical_col': categorical_col,
        'numerical_col': numerical_col,
        'encoder': encoder,
        'new_column_names': new_column_names,
        'feature_names': feature_names,
        'model': model
    }

def save_model(filename, artifacts):
    with open(filename, 'wb') as f_out:
        joblib.dump(artifacts, f_out)

    print(f'artifacts saved to {filename}')

df_full_train = load_data()
artifacts = train_model(df_full_train)
save_model('models/model_artifacts.joblib', artifacts)