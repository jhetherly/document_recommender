import string
import re

from unidecode import unidecode
import requests
from lxml import html
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk.data
nltk.data.path += ['data/nltk']
nltk.download('all', download_dir='data/nltk')
from nltk.tokenize.nist import NISTTokenizer
# from nltk.tokenize.stanford import StanfordTokenizer
from nltk import FreqDist

# nltk.download('punkt')
# nltk.download('perluniprops')


def remove_html_tags_and_brackets(html_text):
    clean_tag = re.compile('<.*?>')
    cleantext = re.sub(clean_tag, ' ', html_text)

    clean_bracket = re.compile('\[.*?\]')
    cleantext = re.sub(clean_bracket, ' ', cleantext)

    clean_parens = re.compile('\(.*?\)')
    cleantext = re.sub(clean_parens, ' ', cleantext)

    return cleantext


def get_valid_lists_pages_and_subcategories(sess, url):
    pages = []
    subcategories = []
    base_url = 'https://en.wikipedia.org'

    page = sess.get(url)
    tree = html.fromstring(page.content)

    pages_content = tree.xpath('//div[@id="bodyContent"]//div[@id="mw-pages"]')
    subcategories_content = tree.xpath('//div[@id="bodyContent"]//div[@id="mw-subcategories"]')

    assert(len(pages_content) <= 1)
    assert(len(subcategories_content) <= 1)

    if len(pages_content) > 0:
        pages_content = pages_content[0]
        for group in pages_content.xpath('//div[@class="mw-category-group"]'):
            label = group.xpath('h3/text()')[0]
            if label not in string.ascii_uppercase:
                continue
            items = group.xpath('ul/li/a/@href')
            for item in items:
                pages.append(base_url + item)

    if len(subcategories_content) > 0:
        subcategories_content = subcategories_content[0]
        for group in subcategories_content.xpath('//div[@class="mw-category-group"]'):
            label = group.xpath('h3/text()')[0]
            if label not in string.ascii_uppercase:
                continue
            skip_group = False
            items = group.xpath('ul/li')
            for item in items:
                spans = group.xpath('div/div/span')
                for span in spans:
                    skip_group |= 'empty' in span.text_content()
                if skip_group:
                    continue
                link = item.xpath('div/div/a/@href')
                if len(link) != 1:
                    continue
                subcategories.append(base_url + str(link[0]))

    return pages, subcategories


def process_page(sess, url, word_tokenizer, lemmatizer=None, sentence_tokenizer=None):
    page = sess.get(url)
    tree = html.fromstring(page.content)

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

    tokenized = []
    if sentence_tokenizer is not None:
        tokenized_sentences = sentence_tokenizer(full_text)
        for tokenized_sentence in tokenized_sentences:
            tokenized_words = word_tokenizer(tokenized_sentence)

            if lemmatizer is not None:
                tokenized_words = [lemmatizer.lemmatize(x) for x in tokenized_words]

            # transliterate to ascii with unidecode and remove non-alpha characters on words of length greater than one
            tokenized_words = [non_lower_alpha.sub('', unidecode(x).lower()) for x in tokenized_words if len(x) > 1]
            tokenized += tokenized_words

    return tokenized

def process_book(sess, url):
    page = sess.get(url)
    tree = html.fromstring(page.content)


def get_wiki_page_data(sess, title):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        'action': "parse",
        'page': title,
        'format': "json"
    }

    page = sess.get(url=url, params=params)
    raw_data = page.json()

    print(raw_data['parse']['text']['*'])


def main():
    topic = 'Physics'
    min_pages = 500
    wiki_url = 'https://en.wikipedia.org/wiki/Category:{}'.format(topic)

    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle').tokenize
    # # word_tokenizer = StanfordTokenizer('data/nltk/stanford-english-corenlp-2018-10-05-models.jar', options={"americanize": True})
    word_tokenizer = NISTTokenizer().tokenize
    lem = nltk.WordNetLemmatizer()

    S = requests.Session()

    # title = "Pet door"

    # get_wiki_page_data(S, title)

    pages, subcategories = get_valid_lists_pages_and_subcategories(S, wiki_url)

    # scape Wikipedia for a given number of pages (with an additional buffer)
    while len(subcategories) > 0 and len(pages) < 1.1*min_pages:
        current_subcategories = subcategories[:]
        for subcategory in current_subcategories:
            subpages, subsubcategories = get_valid_lists_pages_and_subcategories(S, subcategory)
            pages += subpages
            subcategories += subsubcategories
        subcategories = [x for x in subcategories if x not in current_subcategories]

    total_vocab = FreqDist()
    document_vocabs = {}
    for page in pages:
        if 'wiki/Book:' in page:
            vocabs = process_book(S, page, word_tokenizer, lem, sent_tokenizer)
            for book_page in vocabs.keys():
                document_vocabs[book_page] = FreqDist().update(vocabs[book_page])
                total_vocab.update(vocabs[book_page])
        else:
            l = process_page(S, page, word_tokenizer, lem, sent_tokenizer)
            document_vocabs[page] = FreqDist().update(l)
            total_vocab.update(l)
        print(len(dict(total_vocab)))



if __name__ == '__main__':
    main()
