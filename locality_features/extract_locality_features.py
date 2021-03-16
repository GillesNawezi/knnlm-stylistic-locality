from elasticsearch import Elasticsearch
import numpy as np


class ESSearcher:
    def __init__(self, index_name: str):
        self.es = Elasticsearch()
        self.index_name = index_name

    def get_topk(self, query_str: str, field: str, topk: int = 5):
        results = self.es.search(
            index=self.index_name,
            body={'query': {'match': {field: query_str}}})['hits']['hits'][:topk]
        return [(doc['_source'], doc['_score']) for doc in results]


def extract_package_name(filename):
    count = 0
    ess = ESSearcher(index_name='java-source-test')

    with open(filename) as datafile, \
            open(filename + '.original_path', 'w', encoding='utf-8') as out_file:
        for line in datafile:
            # package_name = line.strip().split(';')[0]
            # if not package_name.startswith('<s> package'):
            hits = ess.get_topk(line.strip(), 'source_content', topk=1)
            original_path = hits[0][0]['full_path']
            out_file.write(original_path + '\n')
            count += 1
    print(count)

# extract_package_name('examples/language_model/java/java_test_pre')


def generate_locality_matrix(filename):
    paths = []
    locality = []
    with open(filename) as path_file:
        for line in path_file:
            paths.append(line.rsplit('/', 1)[0])

    for p in paths:
        temp_loc = []
        for t in paths:
            if p == t:
                temp_loc.append(1)
            else:
                temp_loc.append(0)
        locality.append(temp_loc)

    locality = np.array(locality)
    np.save(filename + '.npy', locality)


generate_locality_matrix('examples/language_model/java/java_test_pre.original_path')
generate_locality_matrix('examples/language_model/java/java_validation_pre.original_path')
