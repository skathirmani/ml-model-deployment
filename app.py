# Python backend servers: flask, django, tornado

# In anaconda prompt: pip install genvent
from flask import Flask, render_template, request
from gevent.pywsgi import WSGIServer
import joblib
import pandas as pd

app = Flask(__name__)
model_obj = joblib.load('model_expense_predictor.joblib')


@app.route('/')
def home():
  return render_template('home.html')

@app.route('/calculate_premium', methods=['POST'])
def calculate_expenses():
  age = float(request.args.get('age', 30))
  bmi = float(request.args.get('bmi', 28))
  smoker = request.args.get('smoker', 'no')

  df_customer = pd.DataFrame({'age': [age],
                              'bmi': [bmi],
                              'smoker': [smoker]})

  predicted_expenses = model_obj.predict(df_customer)[0]
  return 'Your predicted medical expenses is %.0f' % predicted_expenses



http_server = WSGIServer(('', 5000), app)
http_server.serve_forever()