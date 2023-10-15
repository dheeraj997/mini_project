import pickle
from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

# Load the trained logistic regression model from the pickle file
with open('model1.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


@app.route("/", methods=['GET', 'POST'])  # Allow both GET and POST requests
def index():
    prediction = None  # Initialize the prediction variable

    if request.method == 'POST':
        values = [request.form[f'value{i}'] for i in range(1, 12)]
        values = [float(value) if value.isnumeric()
                  else 0.0 for value in values]

        # Make a prediction using the model
        v = [np.array(values)]
        prediction = model.predict(v)[0]
    if int(prediction) == int(1):
        status = "Elgible for placements"
    else:
        status = "NOT Elgible for placement work hard"
    return render_template('index.html', prediction=status)


if __name__ == '__main__':
    app.run(debug=True)
