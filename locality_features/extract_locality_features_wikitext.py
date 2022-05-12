import numpy as np
import pathlib

global_path = str(pathlib.Path(__file__).parent.parent.resolve())


def generate_article_locality_matrix(filename):
    articles = []
    locality = []
    with open(filename) as article_file:
        for line in article_file:
            articles.append(line.strip())

    for p in articles:
        temp_loc = []
        for t in articles:
            if p == t:
                temp_loc.append(1)
            else:
                temp_loc.append(0)
        locality.append(temp_loc)

    locality = np.array(locality)
    print(locality.sum())
    np.save(filename + '.npy', locality)


generate_article_locality_matrix(global_path + '/examples/language_model/wikitext103_split/wiki_test_tokens')
generate_article_locality_matrix(global_path + '/examples/language_model/wikitext103_split/wiki_train_tokens')
generate_article_locality_matrix(global_path + '/examples/language_model/wikitext103_split/wiki_valid_tokens')


def generate_section_locality_matrix(split_name):
    test_sections = []
    testtrain_sections = []
    locality = []
    with open('examples/language_model/wikitext103_seg/{}.txt.sec'.format(split_name)) as test_section_file, \
            open('examples/language_model/wikitext103_seg/{}train.txt.sec'.format(split_name)) as testtrain_section_file:
        for line in test_section_file:
            test_sections.append(line.strip())
        for line in testtrain_section_file:
            testtrain_sections.append(line.strip())

    for p in test_sections:
        temp_loc = []
        for t in testtrain_sections:
            if p == t:
                temp_loc.append(1)
            else:
                temp_loc.append(0)
        locality.append(temp_loc)

    locality = np.array(locality).astype('int8')
    np.save('examples/language_model/wikitext103_seg/{}train.txt.sec.npy'.format(split_name), locality)


def generate_domain_locality_matrix(split_name):
    test_domains = []
    testtrain_domains = []
    locality = []
    with open('examples/language_model/wikitext103_seg/{}.txt.dom'.format(split_name)) as test_domain_file, \
            open('examples/language_model/wikitext103_seg/{}train.txt.dom'.format(split_name)) as testtrain_domain_file:
        for line in test_domain_file:
            test_domains.append(set(line.strip().split(';')))
        for line in testtrain_domain_file:
            testtrain_domains.append(set(line.strip().split(';')))

    for p in test_domains:
        temp_loc = []
        for t in testtrain_domains:
            # if has any intersection
            if p.intersection(t):
                temp_loc.append(1)
            else:
                temp_loc.append(0)
        locality.append(temp_loc)

    locality = np.array(locality).astype('int8')
    np.save('examples/language_model/wikitext103_seg/{}train.txt.dom.npy'.format(split_name), locality)


#generate_section_locality_matrix('valid')
#generate_domain_locality_matrix('valid')

