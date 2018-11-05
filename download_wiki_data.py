import string
import requests
from lxml import html


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


def process_page(url):
    pass

def process_book(url):
    pass


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

    for page in pages:
        if 'wiki/Book:' in page:
            process_book(page)
        else:
            process_page(page)
    



if __name__ == '__main__':
    main()