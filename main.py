from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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
@app.route('/predict_all', methods=['GET'])
def predict_all():
    data = pd.read_csv('src/data/data.csv')  # Load your dataset

    # Optionally add YEAR_2 column if needed
    data['YEAR_2'] = data['EXP_YEARS'] + 2

    # Select the features for prediction
    X = data[['EXP_YEARS', 'YEAR_2', 'GENDER', 'MARITAL_STATUS', 'TYPE_DIPLOMA', 'Grade']]

    # Drop rows with any NaN values
    X = X.dropna()

    # Encode categorical features
    label_encoders = {}
    categorical_columns = ['GENDER', 'MARITAL_STATUS', 'TYPE_DIPLOMA', 'Grade']

    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    # Predict the salaries
    predicted_salaries = model.predict(X)

    # Add predictions to the DataFrame
    data = data.loc[X.index]  # Keep only the rows without NaNs
    data['Predicted_Salary'] = predicted_salaries
    total_predicted_salary = predicted_salaries.sum()

    # Return the total sum of predicted salaries
    result = {"Total_Predicted_Salary": total_predicted_salary}
    return jsonify(result)


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
