__author__ = 'mateuszopala'
# coding=utf-8
from main import LanguagePredictor


if __name__ == "__main__":
    language_to_data_mapping_json_path = 'data/language_to_data_mapping.json'
    n = 3
    data_root_path = 'data'
    language_predictor = LanguagePredictor(language_to_data_mapping_json_path, n, data_root_path)
    sentence = "dave, i can't do it i'm afraid"
    corpora = [sentence]
    predicted_language, _, _, _ = language_predictor.predict(corpora)
    print predicted_language