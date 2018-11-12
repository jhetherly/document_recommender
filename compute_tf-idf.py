import os
import string
import argparse
from glob import glob
import json
import urllib

from tqdm import tqdm
import numpy as np
from nltk import FreqDist
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import pairwise_distances


def create_count_vector(doc_vocab, total_vocab_list):
    vec = [0]*len(total_vocab_list)
    for k in doc_vocab.keys():
        if k in total_vocab_list:
            vec[total_vocab_list.index(k)] = doc_vocab[k]
    return vec


def main(download_settings_filename, parse_settings_filename, similarity_settings_filename):
    with open(download_settings_filename, 'r') as f:
        download_config = json.load(f)
    with open(parse_settings_filename, 'r') as f:
        parse_config = json.load(f)
    with open(similarity_settings_filename, 'r') as f:
        similarity_config = json.load(f)
    topic = download_config.get('topic', 'Medicine')
    data_dir = os.path.join(download_config.get('save_dir', os.path.join('data', 'wiki')), topic)
    n_pages = download_config.get('min_pages', 500)
    vocab_dir = os.path.join(parse_config.get('save_dir', os.path.join('artifacts', 'wiki')), topic, 'vocab')
    save_dir = os.path.join(similarity_config.get('save_dir', os.path.join('artifacts', 'wiki')), topic, 'graph')
    vocab_top_k = similarity_config.get('vocab_top_k', 100)
    graph_top_k = similarity_config.get('graph_top_k', 10)
    metric = similarity_config.get('metric', 'euclidean')

    json_files = glob(os.path.join(vocab_dir, '*.json'))

    total_vocab_filename = os.path.join(vocab_dir, 'total_count.json')
    with open(total_vocab_filename, 'r') as f:
        total_vocab = json.load(f)

    total_freq = FreqDist(total_vocab)
    total_number_words = total_freq.N()
    most_freq_words = total_freq.most_common(vocab_top_k)
    percentage_used = 100*sum([x[1] for x in most_freq_words])/total_number_words
    total_vocab_list = [x[0] for x in most_freq_words]
    all_vocabs = []
    good_json_indices = []
    print('reading in preprocessed vocabulary using {:.2f}% of the total count of words'.format(percentage_used))
    i = -1
    for json_file in tqdm(json_files):
        i += 1
        if json_file == total_vocab_filename:
            continue
        with open(json_file, 'r') as f:
            doc_vocab = json.load(f)
            vec = create_count_vector(doc_vocab, total_vocab_list)
            if sum(vec) > 0:
                all_vocabs.append(vec)
                good_json_indices.append(i)
        if len(good_json_indices) >= n_pages:
            continue
    good_json_indices = np.array(good_json_indices)

    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(all_vocabs)

    print('computing similarity matrix')
    similarity_matrix = pairwise_distances(tfidf, metric=metric)

    # NOTE: ignore shortest as this is always "self"
    shortest_indices = np.argsort(similarity_matrix, axis=-1)[:, 1:graph_top_k + 1]

    print('finding top-{} closest pages'.format(graph_top_k))
    pbar = tqdm(total=graph_top_k*min(shortest_indices.shape[0], n_pages))
    analysis_results = {}
    for i in range(graph_top_k):
        ith_shortest = similarity_matrix[np.arange(shortest_indices.shape[0]), shortest_indices[:, i]]
        ith_shortest_indices = shortest_indices[:, i]
        for doc_index in range(min(shortest_indices.shape[0], n_pages)):
            doc_name = os.path.basename(json_files[good_json_indices[doc_index]])
            doc_name = urllib.parse.unquote(doc_name[:doc_name.rfind('.')])
            if doc_name not in analysis_results:
                analysis_results[doc_name] = {"names": [], "similarities": []}
            ith_shortest_doc_name = os.path.basename(json_files[good_json_indices[ith_shortest_indices[doc_index]]])
            ith_shortest_doc_name = urllib.parse.unquote(ith_shortest_doc_name[:ith_shortest_doc_name.rfind('.')])
            analysis_results[doc_name]["names"].append(ith_shortest_doc_name)
            analysis_results[doc_name]["similarities"].append(ith_shortest[doc_index])
            pbar.update(1)
    pbar.close()

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'raw_graph_info.json'), 'w') as f:
        json.dump(analysis_results, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process json files to compute the tf-idf vectors')
    parser.add_argument('--download_settings', type=str,
                        default='settings/wiki_download.json', help='download settings file')
    parser.add_argument('--parse_settings', type=str,
                        default='settings/wiki_parse.json', help='parse settings file')
    parser.add_argument('--similarity_settings', type=str,
                        default='settings/similarity.json', help='similarity settings file')

    args = parser.parse_args()
    main(args.download_settings, args.parse_settings, args.similarity_settings)
