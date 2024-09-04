from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model
model = joblib.load('salary_prediction_model.joblib')


# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')


# Define a route to predict salaries for all employees
@app.route('/predict_all', methods=['GET'])
def predict_all():
    # Example: Load the data from a CSV or database (here assumed from CSV for simplicity)
    data = pd.read_csv('employee_data.csv')

    # Extracting the necessary features
    X = data[['EXP_YEARS', 'YEAR_2', 'GENDER', 'MARITAL_STATUS', 'TYPE_DIPLOMA', 'Grade']]

    # Predict salaries for all employees
    predicted_salaries = model.predict(X)

    # Add predicted salaries to the dataframe
    data['Predicted_Salary'] = predicted_salaries

    # Convert the data to JSON and return it
    return jsonify(data.to_dict(orient='records'))


# Define a route to predict salary for personalized inputs
@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the POST request
    input_data = request.json

    # Convert the input data to a DataFrame
    input_df = pd.DataFrame([input_data])

    # Predict the salary
    predicted_salary = model.predict(input_df)

    # Return the prediction
    return jsonify({'Predicted_Salary': predicted_salary[0]})


if __name__ == "__main__":
    app.run(debug=True)
