import os
import string
import argparse
from glob import glob
import json

from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import pairwise_distances


def create_count_vector(doc_vocab, total_vocab_list):
    vec = [0]*len(total_vocab_list)
    for k in doc_vocab.keys():
        vec[total_vocab_list.index(k)] = doc_vocab[k]
    return vec


def main(download_settings_filename, similarity_settings_filename):
    with open(download_settings_filename, 'r') as f:
        download_config = json.load(f)
    with open(similarity_settings_filename, 'r') as f:
        similarity_config = json.load(f)
    topic = download_config.get('topic', 'Medicine')
    save_dir = os.path.join(download_config.get('save_dir', os.path.join('data', 'wiki')), topic)
    k = similarity_config.get('k', 10)
    metric = similarity_config.get('metric', 'euclidean')

    json_files = glob(os.path.join(save_dir, '*.json'))

    total_vocab_filename = os.path.join(save_dir, 'tf-idf_total.json')
    with open(total_vocab_filename, 'r') as f:
        total_vocab = json.load(f)

    total_vocab_list = list(total_vocab.keys())
    all_vocabs = []
    for json_file in tqdm(json_files):
        if json_file == total_vocab_filename:
            continue
        with open(json_file, 'r') as f:
            doc_vocab = json.load(f)
            vec = create_count_vector(doc_vocab, total_vocab_list)
            all_vocabs.append(vec)

    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(all_vocabs)

    similarity_matrix = pairwise_distances(tfidf, metric='euclidean')
    # similarity_matrix = pairwise_distances(tfidf, metric='cosine')

    # NOTE: ignore shortest as this is always "self"
    shortest_indices = np.argsort(similarity_matrix, axis=-1)[:, 1:k + 1]

    analysis_results = {}
    for i in range(k):
        ith_shortest = similarity_matrix[np.arange(shortest_indices.shape[0]), shortest_indices[:, i]]
        for doc_index in range(shortest_indices.shape[0]):
            doc_name = os.path.basename(json_files[doc_index])
            doc_name = doc_name[:doc_name.rfind('.')]
            if doc_name not in analysis_results:
                analysis_results[doc_name] = {"names": [], "similarities": []}
            ith_shortest_doc_name = os.path.basename(json_files[ith_shortest[doc_index]])
            ith_shortest_doc_name = ith_shortest_doc_name[:ith_shortest_doc_name.rfind('.')]
            analysis_results[doc_name]["names"].append(ith_shortest_doc_name)
            analysis_results[doc_name]["similarities"].append(similarity_matrix[doc_index, ith_shortest[doc_index]])

    print(analysis_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process json files to compute the tf-idf vectors')
    parser.add_argument('--download_settings', type=str,
                        default='settings/wiki_download.json', help='download settings file')
    parser.add_argument('--similarity_settings', type=str,
                        default='settings/similarity.json', help='similarity settings file')

    args = parser.parse_args()
    main(args.download_settings, args.similarity_settings)
