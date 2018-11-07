import os
os.environ["PATH"] += os.pathsep + "C:/ProgramData/Miniconda3/envs/document_recommender/Library/bin/graphviz"
import argparse
import json

from graphviz import Digraph


# import nltk
# print(nltk.__file__)


page_node_name_counter = 0
page_node_name_dict = {}


def create_graphviz_digraph(raw_data, title):
    global page_node_name_counter
    global page_node_name_dict
    dot = Digraph(comment=title)

    for page_name in raw_data.keys():
        similar_pages = raw_data[page_name]['names']
        if page_name not in page_node_name_dict:
            page_node_name_dict[page_name] = str(page_node_name_counter)
            page_node_name_counter += 1
        node_name = page_node_name_dict[page_name]

        dot.node(node_name, page_name)
        for rank, similar_page in enumerate(similar_pages):
            if similar_page not in page_node_name_dict:
                page_node_name_dict[similar_page] = str(page_node_name_counter)
                page_node_name_counter += 1
            similar_node = page_node_name_dict[similar_page]
            dot.node(similar_node, similar_page)
            dot.edge(node_name, similar_node, label=str(rank + 1))

    return dot


def main(download_settings_filename, parse_settings_filename, similarity_settings_filename):
    with open(download_settings_filename, 'r') as f:
        download_config = json.load(f)
    with open(parse_settings_filename, 'r') as f:
        parse_config = json.load(f)
    with open(similarity_settings_filename, 'r') as f:
        similarity_config = json.load(f)
    topic = download_config.get('topic', 'Medicine')
    save_dir = os.path.join(similarity_config.get('save_dir', os.path.join('artifacts', 'wiki')), topic, 'graph')
    graph_top_k = similarity_config.get('graph_top_k', 10)
    graph_title = similarity_config.get('graph_title', 'similarity graph')

    with open(os.path.join(save_dir, 'raw_graph_info.json'), 'r') as f:
        analysis_results = json.load(f)
    
    graph = create_graphviz_digraph(analysis_results, graph_title)

    graph.render(filename=os.path.join(save_dir, 'similarity_graph'))

    graph.format = 'pdf'
    graph.render(filename=os.path.join(save_dir, 'similarity_graph'))


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