from flask import Flask, render_template,request
import pickle
import numpy as np


model = pickle.load(open('modelIshan.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('file.html')

@app.route('/predict', methods=['POST'])
def predict_FoodName():
    Cost_input = np.double(request.form.get('Cost'))
    Calories_input = float(request.form.get('Calories'))

    Cost = Cost_input/1000
    Calories = Calories_input/1000

    result = model.predict(np.array([Cost,Calories]).reshape(1, -1))

    return render_template('file.html', result = str(result))

""" @app.route('/predict', methods=['POST'])
def predict_FoodName():
    Cost = np.double(request.form.get('Cost'))
    Calories = int(request.form.get('Calories'))
    Disease = request.form.get('Disease')

    result = model.predict(np.array([Cost,Calories,Disease])).reshape(-1,1)

    return str(result) """


if __name__ == '__main__':
    app.run(debug = True)


""" from flask import Flask, render_template,request
import pickle
import numpy as np

model = pickle.load(open('modelIshan.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('file.html')

@app.route('/predict', methods=['POST'])
def predict_FoodName():
    Cost = request.form.get('Cost')
    Calories = request.form.get('Calories')
    Disease = request.form.get('Disease')

    if Disease:
        Disease = int(Disease)
    else:
        Disease = np.nan

    result = model.predict(np.array([Cost,Calories,Disease])).reshape(-1,1)

    return str(result)

if __name__ == '__main__':
    app.run(debug = True) """


