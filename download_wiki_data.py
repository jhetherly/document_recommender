import os
import string
import argparse
import json

from tqdm import tqdm
from unidecode import unidecode
import requests
from lxml import html


BASE_URL = 'https://en.wikipedia.org'

def get_valid_lists_pages_and_subcategories(sess, url):
    pages = []
    subcategories = []

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
                pages.append(BASE_URL + item)

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
                    # if this link has no further links skip it (i.e.  it's invalid)
                    skip_group |= 'empty' in span.text_content()
                if skip_group:
                    continue
                link = item.xpath('div/div/a/@href')
                if len(link) != 1:
                    continue
                subcategories.append(BASE_URL + str(link[0]))

    return pages, subcategories


def get_book_pages(sess, url):
    page = sess.get(url)
    tree = html.fromstring(page.content)

    page_links = tree.xpath('//div[@id="mw-content-text"]//dd/a/@href')

    pages = []
    for page_link in page_links:
        pages.append(BASE_URL + page_link)

    return pages


def save_wiki_page(sess, url, save_dir):
    page = sess.get(url)
    filename = os.path.join(save_dir, unidecode(url.split('/')[-1]) + '.html')

    with open(filename, "w") as html_file:
        html_file.write(str(page.content))


def main(settings_filename):
    with open(settings_filename, 'r') as f:
        config = json.load(f)
    topic = config.get('topic', 'Medicine')
    min_pages = config.get('min_pages', 500)
    save_dir = os.path.join(config.get('save_dir', os.path.join('data', 'wiki')), topic)
    buffer = config.get('buffer', 10) # percentage

    wiki_url = 'https://en.wikipedia.org/wiki/Category:{}'.format(topic)

    min_pages_plus_buffer = int(min_pages*(1 + buffer/100))

    print('will scan Wikipedia for {} pages in topic {}'.format(min_pages_plus_buffer, topic))

    S = requests.Session()

    pages, subcategories = get_valid_lists_pages_and_subcategories(S, wiki_url)

    # scape Wikipedia for a given number of pages (with an additional buffer)
    while len(subcategories) > 0 and len(pages) < min_pages_plus_buffer:
        current_subcategories = subcategories[:]
        for subcategory in current_subcategories:
            subpages, subsubcategories = get_valid_lists_pages_and_subcategories(S, subcategory)
            pages += subpages
            subcategories += subsubcategories

            # get all the pages within each book
            book_pages = [x for x in pages if 'wiki/Book:' in x]
            while len(book_pages) > 0 and len(pages) < min_pages_plus_buffer:
                for page in book_pages:
                    pages_in_book = get_book_pages(S, page)
                    pages += pages_in_book
                pages = [x for x in pages if x not in book_pages]
                book_pages = [x for x in pages if 'wiki/Book:' in x]

        subcategories = [x for x in subcategories if x not in current_subcategories]

    os.makedirs(save_dir, exist_ok=True)
    print('saving Wikipedia html files to {}'.format(save_dir))
    for page in tqdm(pages[:min_pages_plus_buffer]):
        save_wiki_page(S, page, save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Wikipedia Category')
    parser.add_argument('--download_settings', type=str,
                        default='settings/wiki_download.json', help='download settings file')

    args = parser.parse_args()
    main(args.download_settings)
