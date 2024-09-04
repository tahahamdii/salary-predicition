from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model
model = joblib.load('salary_prediction_model.joblib')

gender_mapping = {'Male': 0, 'Female': 1}
marital_status_mapping = {'Single': 0, 'Married': 1, 'Divorced': 2}
type_diploma_mapping = {'Bachelor': 0, 'Master': 1, 'PhD': 2}

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')


# Define a route to predict salaries for all employees
@app.route('/predict_all', methods=['GET'])
def predict_all():
    # Example: Load the data from a CSV or database (here assumed from CSV for simplicity)
    data = pd.read_csv('src/data/data.csv')

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
    try:
        data = request.get_json()

        # Create a DataFrame from the input data
        input_df = pd.DataFrame([data])

        # Encode categorical variables
        input_df['GENDER'] = input_df['GENDER'].map(gender_mapping)
        input_df['MARITAL_STATUS'] = input_df['MARITAL_STATUS'].map(marital_status_mapping)
        input_df['TYPE_DIPLOMA'] = input_df['TYPE_DIPLOMA'].map(type_diploma_mapping)

        # Ensure the order of columns matches the training data
        input_df = input_df[['EXP_YEARS', 'YEAR_2', 'GENDER', 'MARITAL_STATUS', 'TYPE_DIPLOMA', 'Grade']]

        # Predict salary
        predicted_salary = model.predict(input_df)

        return jsonify({"Predicted_Salary": predicted_salary[0]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    app.run(debug=True)
