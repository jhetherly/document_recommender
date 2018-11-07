import os
import string
import argparse
from glob import glob
import json
import re

from tqdm import tqdm
from unidecode import unidecode
import requests
from lxml import html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
# from nltk.tokenize import sent_tokenize, word_tokenize
import nltk.data
nltk.data.path += ['data/nltk_utils']
nltk.download('perluniprops', download_dir='data/nltk_utils')
nltk.download('punkt', download_dir='data/nltk_utils')
nltk.download('wordnet', download_dir='data/nltk_utils')
nltk.download('stopwords', download_dir='data/nltk_utils')
from nltk.tokenize.nist import NISTTokenizer
from nltk import FreqDist


stop_words = set(stopwords.words('english'))


def plot_freqdist_freq(fd,
                       max_num=None,
                       cumulative=False,
                       title='Frequency plot',
                       linewidth=2):
    """
    From https://martinapugliese.github.io/plotting-the-actual-frequencies-in-a-FreqDist-in-nltk/
    Work this around here.
    
    INPUT:
        - the FreqDist object
        - max_num: if specified, only plot up to this number of items 
          (they are already sorted descending by the FreqDist)
        - cumulative: bool (defaults to False)
        - title: the title to give the plot
        - linewidth: the width of line to use (defaults to 2)
    OUTPUT: plot the freq and return None.
    """

    tmp = fd.copy()
    norm = fd.N()
    for key in tmp.keys():
        tmp[key] = float(fd[key]) / norm

    if max_num:
        tmp.plot(max_num, cumulative=cumulative,
                 title=title, linewidth=linewidth)
    else:
        tmp.plot(cumulative=cumulative, 
                 title=title, 
                 linewidth=linewidth)


def save_freq_plot(save_filename, freq, **kw_args):
    plot_freqdist_freq(freq, **kw_args)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if os.path.isfile(save_filename):
        os.remove(save_filename)
    plt.savefig(save_filename)
    plt.close()


def remove_html_tags_and_brackets(html_text):
    clean_tag = re.compile('<.*?>')
    cleantext = re.sub(clean_tag, ' ', html_text)

    clean_bracket = re.compile('\[.*?\]')
    cleantext = re.sub(clean_bracket, ' ', cleantext)

    clean_parens = re.compile('\(.*?\)')
    cleantext = re.sub(clean_parens, ' ', cleantext)

    return cleantext


def process_page(sess, filename, exclude_list, word_tokenizer, lemmatizer=None, sentence_tokenizer=None):
    with open(filename) as f:
        tree = html.fromstring(f.read())

    content = tree.xpath('//div[@id="mw-content-text"]')

    assert(len(content) <= 1)

    if len(content) == 1:
        content = content[0]
    else:
        return

    full_text = ''
    for paragraph in content.xpath('div[@class="mw-parser-output"]/p'):
        math_stuff = paragraph.xpath('child::span[@class="mwe-math-element"]')
        for ms in math_stuff:
            paragraph.remove(ms)
        citation_stuff = paragraph.xpath('child::sup[@class="reference"]')
        for cs in citation_stuff:
            paragraph.remove(cs)
        text = str(paragraph.text_content())
        text = remove_html_tags_and_brackets(text)
        full_text += text

    non_lower_alpha = re.compile('[^a-z]')

    def tokenize_words(t):
        tokenized_words = word_tokenizer(t)

        if lemmatizer is not None:
            tokenized_words = [lemmatizer.lemmatize(x) for x in tokenized_words]

        # transliterate to ascii with unidecode and remove non-alpha characters on words
        tokenized_words = [non_lower_alpha.sub('', unidecode(x).lower()) for x in tokenized_words]
        # remove stop words, etc.
        tokenized_words = [x for x in tokenized_words
                           if x not in stop_words and x not in exclude_list and len(x) > 2]
        return tokenized_words

    tokenized = []
    if sentence_tokenizer is not None:
        tokenized_sentences = sentence_tokenizer(full_text)
        for tokenized_sentence in tokenized_sentences:
            tokenized += tokenize_words(tokenized_sentence)
    else:
        tokenized += tokenize_words(full_text)

    return tokenized


def main(download_settings_filename, parse_settings_filename):
    with open(download_settings_filename, 'r') as f:
        download_config = json.load(f)
    with open(parse_settings_filename, 'r') as f:
        parse_config = json.load(f)
    topic = download_config.get('topic', 'Medicine')
    data_dir = os.path.join(download_config.get('save_dir', os.path.join('data', 'wiki')), topic)
    save_dir = os.path.join(parse_config.get('save_dir', os.path.join('artifacts', 'wiki')), topic, 'vocab')
    exclude_vocab = parse_config.get('exclude_vocab', [])
    min_page_vocab = parse_config.get('min_page_vocab', 5)
    plot_top_k = parse_config.get('plot_top_k', 40)
    plot_cumulative = parse_config.get('plot_cumulative', True)
    plot_title = 'top {} frequency'.format(plot_top_k) if not plot_cumulative else 'top {} cumulative'.format(plot_top_k)
    make_plots = plot_top_k > 0

    wiki_url = 'https://en.wikipedia.org/wiki/Category:{}'.format(topic)

    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle').tokenize
    word_tokenizer = NISTTokenizer().tokenize
    lem = nltk.WordNetLemmatizer()

    S = requests.Session()

    pages = glob(os.path.join(data_dir, '*.html'))

    total_vocab = FreqDist()
    document_vocabs = {}
    print('reading {} files and generating vocabulary'.format(len(pages)))
    os.makedirs(save_dir, exist_ok=True)
    for page in tqdm(pages):
        l = process_page(S, page, exclude_vocab, word_tokenizer, lem, sent_tokenizer)
        # ignore pages with very small vocabulary
        if len(l) < min_page_vocab:
            continue
        document_vocabs[page] = FreqDist(l)
        total_vocab.update(l)
        save_filename = os.path.join(save_dir, os.path.basename(page[:page.rfind('.')]) + '.json')
        with open(save_filename, 'w') as f:
            json.dump(dict(document_vocabs[page]), f)
        if make_plots:
            save_filename = save_filename[:save_filename.rfind('.')] + '.pdf'
            save_freq_plot(save_filename, document_vocabs[page], max_num=plot_top_k, cumulative=plot_cumulative, title=plot_title)
    with open(os.path.join(save_dir, 'total_count.json'), 'w') as f:
        json.dump(dict(total_vocab), f)
    if make_plots:
        save_filename = os.path.join(save_dir, 'total_count.pdf')
        save_freq_plot(save_filename, total_vocab, max_num=plot_top_k, cumulative=plot_cumulative, title=plot_title)
    # print(total_vocab.most_common(100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process html files to build a vocabulary for tf-idf analysis')
    parser.add_argument('--download_settings', type=str,
                        default='settings/wiki_download.json', help='download settings file')
    parser.add_argument('--parse_settings', type=str,
                        default='settings/wiki_parse.json', help='parse settings file')

    args = parser.parse_args()
    main(args.download_settings, args.parse_settings)
