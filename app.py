from __future__ import annotations

import os
import subprocess
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, redirect, render_template, request, url_for
from flask_wtf import CSRFProtect
from flask_wtf.csrf import CSRFError, generate_csrf

MODEL_FILE_PATH = Path("trips_data.joblib")
MODEL_URL = "https://github.com/rksi-man/datasets/raw/main/data/trips_data.joblib"
EXPECTED_ENCODER_KEYS = ("city", "vacation_preference", "transport_preference")
LEGACY_ENCODER_ALIASES = {
    "preference": "vacation_preference",
    "transport": "transport_preference",
}


class UnknownCategoryError(ValueError):
    """Raised when incoming categorical values are unknown to the encoder."""


def ensure_model_file_exists(model_file_path: Path, model_url: str) -> None:
    if model_file_path.is_file():
        print(f"Файл '{model_file_path}' уже существует.")
        return

    print(f"Файл '{model_file_path}' не найден. Начинаю загрузку.")
    try:
        urllib.request.urlretrieve(model_url, model_file_path)
    except Exception:
        curl_result = subprocess.run(
            ["curl", "-fL", model_url, "-o", str(model_file_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if curl_result.returncode != 0:
            raise RuntimeError(
                "Ошибка при загрузке файла модели "
                f"{model_file_path}: {curl_result.stderr.strip()}"
            )

    print(f"Файл успешно загружен по ссылке: {model_url}")


def _extract_item(payload: Any, key: str) -> Any:
    if isinstance(payload, Mapping):
        if key in payload:
            return payload[key]
    elif hasattr(payload, key):
        return getattr(payload, key)
    raise KeyError(f"В объекте модели нет обязательного ключа/атрибута '{key}'.")


def _normalize_label_encoders(raw_label_encoders: Any) -> dict[str, Any]:
    encoders: dict[str, Any] = {}

    if isinstance(raw_label_encoders, Mapping):
        encoders.update(raw_label_encoders)
    else:
        for name in (*EXPECTED_ENCODER_KEYS, *LEGACY_ENCODER_ALIASES):
            if hasattr(raw_label_encoders, name):
                encoders[name] = getattr(raw_label_encoders, name)

    for legacy_key, canonical_key in LEGACY_ENCODER_ALIASES.items():
        if legacy_key in encoders and canonical_key not in encoders:
            encoders[canonical_key] = encoders[legacy_key]

    missing = [key for key in EXPECTED_ENCODER_KEYS if key not in encoders]
    if missing:
        raise KeyError(
            f"В модели отсутствуют LabelEncoder для колонок: {', '.join(missing)}"
        )

    return {key: encoders[key] for key in EXPECTED_ENCODER_KEYS}


def load_model_bundle(model_file_path: Path) -> tuple[Any, dict[str, Any]]:
    payload = joblib.load(model_file_path)
    model = _extract_item(payload, "model")
    raw_label_encoders = _extract_item(payload, "label_encoders")
    label_encoders = _normalize_label_encoders(raw_label_encoders)
    return model, label_encoders


def preprocess_input(new_data: pd.DataFrame) -> pd.DataFrame:
    encoded_data = new_data.copy()
    for column in EXPECTED_ENCODER_KEYS:
        encoder = label_encoders[column]
        try:
            encoded_data[column] = encoder.transform(encoded_data[column])
        except ValueError as error:
            allowed_values = ", ".join(map(str, encoder.classes_))
            raise UnknownCategoryError(
                f"Недопустимое значение для '{column}'. "
                f"Ожидалось одно из: {allowed_values}"
            ) from error
    return encoded_data


pd.options.display.float_format = "{:.0%}".format
app = Flask(__name__)
app.config["WTF_CSRF_TIME_LIMIT"] = 500
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-change-me")
csrf = CSRFProtect(app)

ensure_model_file_exists(MODEL_FILE_PATH, MODEL_URL)
loaded_model, label_encoders = load_model_bundle(MODEL_FILE_PATH)


@app.route("/", methods=["GET"])
def index():
    csrf_token = generate_csrf()
    city_options = label_encoders["city"].classes_.tolist()
    vacation_preference_options = label_encoders[
        "vacation_preference"
    ].classes_.tolist()
    transport_preference_options = label_encoders[
        "transport_preference"
    ].classes_.tolist()

    return render_template(
        "index.html",
        csrf_token=csrf_token,
        city_options=city_options,
        vacation_preference_options=vacation_preference_options,
        transport_preference_options=transport_preference_options,
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return redirect(url_for("index"))

    salary = int(request.form["salary"])
    city = request.form["city"]
    age = int(request.form["age"])
    vacation_preference = request.form["vacation_preference"]
    transport_preference = request.form["transport_preference"]
    family_members = int(request.form["family_members"])

    new_data = pd.DataFrame(
        {
            "salary": [salary],
            "city": [city],
            "age": [age],
            "vacation_preference": [vacation_preference],
            "transport_preference": [transport_preference],
            "family_members": [family_members],
        }
    )

    encoded_data = preprocess_input(new_data)
    prediction = loaded_model.predict(encoded_data)
    probability = loaded_model.predict_proba(encoded_data)

    data = {"Class": loaded_model.classes_, "Probability": probability[0]}
    df = pd.DataFrame(data)
    df.sort_values(by="Probability", ascending=False, inplace=True)

    return render_template(
        "result.html",
        prediction=prediction[0],
        probability=f"{probability[0].max() * 100:.0f}",
        detailed_result=df.to_html(index=False),
    )


@app.errorhandler(UnknownCategoryError)
def handle_unknown_category_error(error: UnknownCategoryError):
    return str(error), 400


@app.errorhandler(CSRFError)
def handle_csrf_error(error: CSRFError):  # pylint: disable=unused-argument
    return render_template("csrf_error.html"), 400
