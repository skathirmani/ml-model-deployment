import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
import expenses_model as em
from sklearn.metrics import r2_score


if __name__ == '__main__':
  #print('This file is treated as script')
  #url = 'https://raw.githubusercontent.com/skathirmani/datasets/main/insurance.csv'
  data = pd.read_csv('insurance.csv')
  target_col_name = 'expenses'
  input_col_names = ['age', 'bmi', 'smoker']
  train_x, test_x, train_y, test_y = train_test_split(data[input_col_names],
                                                      data[target_col_name],
                                                      test_size=0.2,
                                                      random_state=1)
  model_obj = em.ExpensesPredictor(train_x, train_y).fit()
  test_y_pred = model_obj.predict(test_x)
  print('Model is built with R2 score:' , r2_score(test_y, test_y_pred))
  joblib.dump(model_obj, 'model_expense_predictor.joblib')                                     