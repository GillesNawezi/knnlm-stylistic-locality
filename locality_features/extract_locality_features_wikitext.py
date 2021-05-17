import numpy as np


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


# generate_article_locality_matrix('../examples/language_model/wikitext103_seg/test.txt.docid')
# generate_subdir_locality_matrix('examples/language_model/java/java_validation_pre.original_path')


def generate_section_locality_matrix():
    test_sections = []
    testtrain_sections = []
    locality = []
    with open('examples/language_model/wikitext103_seg/test.txt.sec') as test_section_file, \
            open('examples/language_model/wikitext103_seg/testtrain.txt.sec') as testtrain_section_file:
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
    np.save('examples/language_model/wikitext103_seg/testtrain.txt.sec.npy', locality)


def generate_domain_locality_matrix():
    test_domains = []
    testtrain_domains = []
    locality = []
    with open('examples/language_model/wikitext103_seg/test.txt.dom') as test_domain_file, \
            open('examples/language_model/wikitext103_seg/testtrain.txt.dom') as testtrain_domain_file:
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
    np.save('examples/language_model/wikitext103_seg/testtrain.txt.dom.npy', locality)


generate_section_locality_matrix()
generate_domain_locality_matrix()

