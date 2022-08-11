import pickle
import re
import tqdm
from typing import List, Tuple
import sling
import time
import os
from collections import defaultdict
import numpy as np

"""
This Script generates the domain and section files for the knn-Lm Dstore
"""
folder = "wikitext103_split"

class SlingExtractor(object):
    def load_kb(self, root_dir: str = 'local/data/e/wiki'):
        print('loading and indexing kb ...')
        start = time.time()
        self.kb = sling.Store()
        self.kb.load(os.path.join(root_dir, 'kb.sling'))
        x=y
        self.phrase = sling.PhraseTable(self.kb, os.path.join(root_dir, 'en', 'phrase-table.repo'))
        self.kb.freeze()
        self.extract_property_names()
        print('loading took', (time.time() - start), 'secs')

    def extract_property_names(self):
        print('storing property names ...')
        start = time.time()
        self.property_names = defaultdict(list)
        for frame in self.kb:
            if 'id' in frame and frame.id.startswith('P'):
                self.property_names[frame.id].append(frame.name)
        print('found', str(len(self.property_names)), 'properties')
        print('took', (time.time() - start), 'sec')

    @staticmethod
    def get_frame_id(frame):
        if 'id' in frame:
            return frame.id
        if 'is' in frame:
            if type(frame['is']) != sling.Frame:
                return None
            if 'id' in frame['is']:
                return frame['is'].id
        return None

    @staticmethod
    def get_date_property(prop, tail):
        if 'target' not in prop:
            return None
        if prop.target.id != '/w/time':
            return None
        prop_id = SlingExtractor.get_frame_id(prop)
        if type(tail) == int:
            return (prop_id, tail)
        elif type(tail) == sling.Frame and 'is' in tail and type(tail['is']) == int:
            return (prop_id, tail['is'])
        return None

    @staticmethod
    def get_canonical_property(prop, tail):
        if type(prop) != sling.Frame or type(tail) != sling.Frame:
            return None
        prop_id = SlingExtractor.get_frame_id(prop)
        tail_id = SlingExtractor.get_frame_id(tail)
        if prop_id is None:
            return None
        if tail_id is None:
            return None
        if not prop_id.startswith('P') or not tail_id.startswith('Q'):
            return None
        return (prop_id, tail_id)

    def get_type(self, wid) -> str:
        for type_prop in ['P31', 'P279']:
            try:
                return self.kb[self.kb[wid][type_prop].id].name
            except:
                pass
        return None

    def get_name(self, wid) -> str:
        return self.kb[wid].name

    def iter_property(self, wid: str, type: str = 'can', shuffle: bool = False):
        tup_li: List[Tuple] = []
        if self.kb[wid] is None:
            return []
        for prop, tail in self.kb[wid]:
            tup = self.get_canonical_property(prop, tail)
            if tup is not None and type == 'can':
                if not hasattr(self, 'filter') or tup[0] in self.filter:
                    tup_li.append(tup)
                continue
            if tup is None:
                tup = self.get_date_property(prop, tail)
                if tup is not None and type == 'time':
                    if not hasattr(self, 'filter') or tup[0] in self.filter:
                        tup_li.append(tup)
                    continue
        group = defaultdict(list)
        for k, v in tup_li:
            group[k].append(v)
        result = list(group.items())
        if shuffle:
            np.random.shuffle(result)
        return result


def load_dbpedia_mapping():
    mapping = {}
    with open('examples/language_model/wikitext103_seg/instance-types_specific.ttl', encoding='utf-8') as mapping_f:
        for line in mapping_f:
            ls = line.strip().split()
            if len(ls) == 4:
                qid = ls[0].rstrip('>').split('/')[-1]
                category = ls[2].rstrip('>').split('/')[-1]
                mapping[qid] = category
    return mapping


# mapping = load_dbpedia_mapping()
se = SlingExtractor()
se.load_kb(root_dir='/home/gilles/sling_data')

def process_splits(pkl_path):
    data = pickle.load(open(pkl_path, 'rb'))
    SECTION_PATTERN = re.compile(r"= = [^=]+ = =")
    segmented_texts = []
    corresponding_doc_ids = []
    section_names = []
    domain_types = []
    domain_not_found_count = 0
    for doc in tqdm.tqdm(data):
        text = ' '.join(doc[0])
        doc_id = doc[1][0][1]
        starts = [0]
        section_match = re.finditer(SECTION_PATTERN, text)
        for m in section_match:
            section_names.append(m.group(0))
            domains = []
            ps = se.iter_property(doc_id, type='can')
            for p in ps:
                if p[0] in {'P31', 'P279'}:
                    domains.extend(p[1])
            if len(domains) == 0:
                domain_not_found_count += 1
            domain_types.append(';'.join(domains))
            starts.append(m.start())
            segmented_texts.append(text[starts[-2]:starts[-1]])
            corresponding_doc_ids.append(doc_id)

    print('domain not found:', domain_not_found_count)
    open(pkl_path.rstrip('.pkl') + '.txt', 'w', encoding='utf-8').write('\n'.join(segmented_texts))
    open(pkl_path.rstrip('.pkl') + '.txt.docid', 'w', encoding='utf-8').write(
        '\n'.join(corresponding_doc_ids))
    open(pkl_path.rstrip('.pkl') + '.txt.sec', 'w', encoding='utf-8').write(
        '\n'.join(section_names))
    open(pkl_path.rstrip('.pkl') + '.txt.dom', 'w', encoding='utf-8').write(
        '\n'.join(domain_types))


if __name__ == '__main__':
    process_splits('examples/language_model/wikitext103_seg/test.pkl')
    process_splits('examples/language_model/wikitext103_seg/valid.pkl')
    process_splits('examples/language_model/wikitext103_seg/train.pkl')
