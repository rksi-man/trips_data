from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd
import os
import urllib.request
from flask_wtf import CSRFProtect
from flask_wtf.csrf import CSRFError, generate_csrf


file_path = 'trips_data.joblib'
url = 'https://github.com/rksi-man/datasets/raw/main/data/trips_data.joblib'

# Проверка наличия файла
if not os.path.isfile(file_path):
    print(f"Файл '{file_path}' не найден. Начинаю загрузку.")

    # Загрузка файла
    try:
        urllib.request.urlretrieve(url, file_path)
        print(f"Файл успешно загружен по ссылке: {url}")
    except Exception as e:
        print(f"Ошибка при загрузке файла: {e}")
else:
    print(f"Файл '{file_path}' уже существует.")


pd.options.display.float_format = '{:.0%}'.format
app = Flask(__name__)
app.config['WTF_CSRF_TIME_LIMIT'] = 500
app.secret_key = 'my_super_puper_key'
csrf = CSRFProtect(app)

loaded_model_data = joblib.load(file_path)
loaded_model = loaded_model_data.model


def preprocess_input(new_data):
    new_data['city'] = loaded_model_data.label_encoders.city.transform(
        new_data['city']
    )
    new_data['vacation_preference'] = loaded_model_data.label_encoders.vacation_preference.transform(
        new_data['vacation_preference']
    )
    new_data['transport_preference'] = loaded_model_data.label_encoders.transport_preference.transform(
        new_data['transport_preference']
    )
    return new_data


@app.route('/', methods=['GET'])
@csrf.exempt
def index():
    csrf_token = generate_csrf()
    label_encoders = loaded_model_data.label_encoders
    city_options = label_encoders.city.classes_.tolist()
    vacation_preference_options = label_encoders.vacation_preference.classes_.tolist(
    )
    transport_preference_options = label_encoders.transport_preference.classes_.tolist(
    )

    return render_template('index.html',
                           csrf_token=csrf_token,
                           city_options=city_options,
                           vacation_preference_options=vacation_preference_options,
                           transport_preference_options=transport_preference_options)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return redirect(url_for('index'))

    if request.method == 'POST':
        salary = int(request.form['salary'])
        city = request.form['city']
        age = int(request.form['age'])
        vacation_preference = request.form['vacation_preference']
        transport_preference = request.form['transport_preference']
        family_members = int(request.form['family_members'])
        print(salary, city, age, vacation_preference,
              transport_preference, family_members)
        new_data = pd.DataFrame({
            'salary': [salary],
            'city': [city],
            'age': [age],
            'vacation_preference': [vacation_preference],
            'transport_preference': [transport_preference],
            'family_members': [family_members]
        })

        new_data = preprocess_input(new_data)

        prediction = loaded_model.predict(new_data)
        probability = loaded_model.predict_proba(new_data)

        data = {'Class': loaded_model.classes_,
                'Probability': probability[0]}
        df = pd.DataFrame(data)
        df.sort_values(by='Probability', ascending=False, inplace=True)

        return render_template('result.html',
                               prediction=prediction[0],
                               probability=f'{probability[0].max() * 100: .0f}',
                               detailed_result=df.to_html(index=False))


@app.errorhandler(CSRFError)
def handle_csrf_error(e):
    return render_template('csrf_error.html'), 400
    # return 'Token expired', 400


