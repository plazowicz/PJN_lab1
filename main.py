# coding=utf-8
__author__ = 'mateuszopala'
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import cPickle


def plot_curve(similarities_per_language, target_language, output_path):
    plt.figure(figsize=(9, 9))
    for language, similarities in similarities_per_language.iteritems():
        plt.plot(np.array(similarities), label='%s' % language)
    plt.legend(fontsize=20)
    plt.title("Target: %s" % target_language)
    plt.xlabel('N-gram size', fontsize=20)
    plt.ylabel('Similarity', fontsize=20)
    plt.savefig(output_path)


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
        languages = []
        similarities = []
        for language, model in self.models_per_language.iteritems():
            x = model.transform(corpora)
            similarity = cosine_similarity(self.representation_per_language[language], x)[0, 0]
            languages.append(language)
            similarities.append(similarity)
            print "Similarity for %s is %f" % (language, similarity)
            if similarity > best_similarity:
                best_similarity = similarity
                predicted_language = language
        return predicted_language, best_similarity, languages, similarities


if __name__ == "__main__":
    language_to_data_mapping_json_path = 'data/language_to_data_mapping.json'
    data_root_path = 'data'

    n_range = list(xrange(0, 7))
    curves_directory = 'data/curves'
    test_directory = 'data/test'

    dump_data_path = 'data/data.pkl'

    data_per_test_file = {}
    for test_file in os.listdir(test_directory):
        target_language = test_file.split('.')[0]
        data_per_test_file[target_language] = defaultdict(list)

    for n in n_range:
        language_predictor = LanguagePredictor(language_to_data_mapping_json_path, n, data_root_path)
        for test_file in os.listdir(test_directory):
            test_path = os.path.join(test_directory, test_file)
            with open(test_path, 'r') as f:
                corpora = f.read()
            target_language = test_file.split('.')[0]
            predicted_language, _, languages, similarities = language_predictor.predict([corpora])
            for language, similarity in zip(languages, similarities):
                data_per_test_file[target_language][language].append(similarity)

    with open(dump_data_path, 'w') as f:
        cPickle.dump(data_per_test_file, f)

    # with open(dump_data_path, 'r') as f:
    #     data_per_test_file = cPickle.load(f)

    for target_language, similarities_per_language in data_per_test_file.iteritems():
        output_path = os.path.join(curves_directory, '%s.jpg' % target_language)
        plot_curve(similarities_per_language, target_language, output_path)
