import json
import os
import re

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

es = Elasticsearch()

print(es.indices.create(index='java-source-test', ignore=400))


def process_original_source(dirname):
    for root, subdirs, files in os.walk(dirname):
        for file in files:
            full_path = os.path.join(root, file)
            source_content = open(full_path).read()
            source_content = source_content.replace('<comment>', '')
            yield source_content, full_path


def gendata(dirname):
    for src, full_path in process_original_source(dirname):
        result = {
            "_index": "java-source-test",
            "_type": "_doc",
            'source_content': src,
            'full_path': full_path
        }
        yield result


all_docs = list(gendata('../examples/language_model/java/Test-processed'))

print(bulk(es, all_docs, index="java-source-test"))
