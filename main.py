# coding=utf-8
__author__ = 'mateuszopala'
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os


def read_json(file_path):
    with open(file_path, 'r') as f:
        obj = json.load(f)
    return obj


def read_corpora(file_path):
    with open(file_path, 'r') as f:
        corpora = f.read().decode('cp1252')
    return corpora


class LanguagePredictor(object):
    def __init__(self, language_to_data_mapping_json_path, n, data_root_path):
        self.models_per_language = {}
        self.representation_per_language = {}
        language_to_data_mapping = read_json(language_to_data_mapping_json_path)
        for language, corporas_files_list in language_to_data_mapping.iteritems():
            print "Processing language: %s" % language
            corporas = []
            for corpora_file_name in corporas_files_list:
                corpora_path = os.path.join(data_root_path, corpora_file_name)
                corporas.append(read_corpora(corpora_path))
            model = CountVectorizer(analyzer='char', ngram_range=(n, n))
            self.representation_per_language[language] = model.fit_transform(corporas).sum(axis=0)
            self.models_per_language[language] = model

    def predict(self, corpora):
        best_similarity = 0.
        predicted_language = None
        for language, model in self.models_per_language.iteritems():
            x = model.transform(corpora)
            similarity = cosine_similarity(self.representation_per_language[language], x)
            print "Similarity for %s is %f" % (language, similarity)
            if similarity > best_similarity:
                best_similarity = similarity
                predicted_language = language
        return predicted_language


if __name__ == "__main__":
    language_to_data_mapping_json_path = 'data/language_to_data_mapping.json'
    n = 2
    data_root_path = 'data'
    language_predictor = LanguagePredictor(language_to_data_mapping_json_path, n, data_root_path)
    sentence = "nosz kurwa mać"
    print language_predictor.predict([sentence])
