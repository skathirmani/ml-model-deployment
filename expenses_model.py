import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

class ExpensesPredictor:
  def __init__(self, train_x, train_y):
    self.org_input_columns = train_x.columns.tolist()
    self.train_x = train_x
    self.train_y = train_y
    self.data_preprocessed = self.preprocess(train_x, is_train_data=True)
  
  def preprocess(self, input_data, is_train_data=False):
    if is_train_data:
      dummies = pd.get_dummies(input_data, drop_first=True)
      dummies = dummies.rename(columns={'smoker_yes': 'is_smoker',
                                        'smoker_no': 'is_smoker'})
      self.encoded_input_cols = dummies.columns
      scaler = StandardScaler().fit(dummies)
      self.scaler = scaler
      dummies_std = scaler.transform(dummies)
    else:
      if len(input_data) == 1:
        dummies = pd.get_dummies(input_data)
      else:
        dummies = pd.get_dummies(input_data, drop_first=True)
      dummies = dummies.rename(columns={'smoker_yes': 'is_smoker',
                                        'smoker_no': 'is_smoker'})
      dummies_std = self.scaler.transform(dummies)
    return dummies_std
  
  def fit(self):
    self.model = LinearRegression().fit(self.data_preprocessed, self.train_y)
    return self

  def predict(self, test_x):
    test_x_std = self.preprocess(test_x, is_train_data=False)
    predicted_values = self.model.predict(test_x_std)
    return predicted_values