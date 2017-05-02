from collections import defaultdict
from dateutil.parser import parse
import pickle
import os
import sys
import tempfile
import shutil

from .corpus_doc_manager import CorpusDocListMgr, CorpusDocFileMgr

class Corpus:
    def __init__(self, lang='en', docs=[], big_ass_data=True, corpus_dir=""):
        # We ignore "lang" in this corpus implementation because we are not
        # industrial strength

        if big_ass_data:
            self.docs = CorpusDocFileMgr(corpus_dir, clean_on_exit=False)
        else:
            self.docs = CorpusDocListMgr()

        i = 0
        for doc in docs:
            self.docs.add_doc(doc)
            i += 1

        self.n_docs = i

    def __getitem__(self, idx_or_slice):
        return self.docs[idx_or_slice]

    def __len__(self):
        return self.n_docs

    def __repr__(self):
        return 'Corpus({} docs)'.format(self.n_docs)

    def __iter__(self):
        for doc in self.docs:
            yield doc

    def get_filename(path, name=""):
        DOC_NAME_SUFFIX = "-industrial_strength_corpus.dat"
        name = name + DOC_NAME_SUFFIX
        corpus_path = os.path.abspath(os.path.join(path, name))
        return corpus_path

    def save(self, path, name=""):
        corpus_path = Corpus.get_filename(path, name)
        with open(corpus_path, "wb+") as corpus_file:
            pickle.dump(self, corpus_file)

    def load(path, name=""):
        corpus_path = Corpus.get_filename(path, name)
        corpus = None
        with open(corpus_path, "rb+") as corpus_file:
            corpus = pickle.load(corpus_file)
        return corpus
