from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

model = joblib.load('salary_prediction_model.joblib')

gender_mapping = {'Male': 0, 'Female': 1}
marital_status_mapping = {'Single': 0, 'Married': 1, 'Divorced': 2}
type_diploma_mapping = {'Bachelor': 0, 'Master': 1, 'PhD': 2}

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict_all', methods=['GET'])
def predict_all():
    data = pd.read_csv('src/data/data.csv')

    data['YEAR_2'] = data['EXP_YEARS'] + 2

    X = data[['EXP_YEARS', 'YEAR_2', 'GENDER', 'MARITAL_STATUS', 'TYPE_DIPLOMA', 'Grade']]

    X = X.dropna()

    label_encoders = {}
    categorical_columns = ['GENDER', 'MARITAL_STATUS', 'TYPE_DIPLOMA', 'Grade']

    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    # Predict the salaries
    predicted_salaries = model.predict(X)

    data = data.loc[X.index]
    data['Predicted_Salary'] = predicted_salaries
    total_predicted_salary = predicted_salaries.sum()

    result = {"Total_Predicted_Salary": total_predicted_salary}
    return jsonify(result)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        input_df = pd.DataFrame([data])

        input_df['GENDER'] = input_df['GENDER'].map(gender_mapping)
        input_df['MARITAL_STATUS'] = input_df['MARITAL_STATUS'].map(marital_status_mapping)
        input_df['TYPE_DIPLOMA'] = input_df['TYPE_DIPLOMA'].map(type_diploma_mapping)

        input_df = input_df[['EXP_YEARS', 'YEAR_2', 'GENDER', 'MARITAL_STATUS', 'TYPE_DIPLOMA', 'Grade']]

        predicted_salary = model.predict(input_df)

        return jsonify({"Predicted_Salary": predicted_salary[0]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    app.run(debug=True)
