import os
import sys
import tempfile
import shutil

class CorpusDocMgr:
    def __init__(self):
        pass

    def add_doc(self, doc):
        raise NotImplementedError

    def __getitem__(self, val):
        raise NotImplementedError

    def __delitem__(self, val):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __del__(self):
        return

class CorpusDocListMgr(CorpusDocMgr):
    def __init__(self):
        CorpusDocMgr.__init__(self)
        self.docs = []

    def add_doc(self, doc):
        self.docs.append(doc)

    def __getitem__(self, val):
        return self.docs.__getitem__(val)

    def __delitem__(self, val):
        del self.docs[val]
        return

    def __iter__(self):
        for doc in self.docs:
            yield doc

    def __len__(self):
        return len(self.docs)

class CorpusDocFileMgr(CorpusDocMgr):
    def __init__(self, corpus_dir=""):
        CorpusDocMgr.__init__(self)

        self.DOC_PREFIX = "corpus-doc-{0}-"

        if corpus_dir == "":
            self.temp_docs_dir = tempfile.mkdtemp()
        else:
            self.temp_docs_dir = corpus_dir
        self.next_doc_index = 0
        self.doc_index_to_hash = {}

        print("CREATED CORPUS DOC MGR AT {0}".format(self.temp_docs_dir))

    def add_doc(self, doc):
        doc_hash = str(hash(doc))
        self.doc_index_to_hash[self.next_doc_index] = doc_hash
        doc.save(self.temp_docs_dir, name=self.DOC_PREFIX.format(doc_hash))
        self.next_doc_index += 1

    def __getitem__(self, idx):
        def load_doc(doc_index):
            doc_hash = self.doc_index_to_hash[doc_index]
            return Document.load(self.temp_docs_dir, name=self.DOC_PREFIX.format(doc_hash))

        if type(idx) is slice:
            start = idx.start
            if start is None:
                start = 0

            stop = idx.stop
            if stop is None:
                stop = self.next_doc_index
            elif stop < 0:
                stop = self.next_doc_index + stop + 1

            if start < 0 or start >= self.next_doc_index:
                raise KeyError

            if stop < 0 or stop > self.next_doc_index:
                raise KeyError

            doc_list = []
            for doc_index in range(start, stop, 1 if idx.step is None else idx.step):
                doc_list.append(load_doc(doc_index))
            return doc_list
        else:
            if idx < 0:
                idx = self.next_doc_index + idx

            if idx < 0 or idx >= self.next_doc_index:
                raise KeyError

            return load_doc(idx)

    def __delitem__(self, index_or_splice):
        def remove_doc(doc_index):
            doc_hash = self.doc_index_to_hash[doc_index]
            for index_to_shift in range(doc_index + 1, self.next_doc_index):
                self.doc_index_to_hash[index_to_shift - 1] = \
                        self.doc_index_to_hash[index_to_shift]

        if type(index_or_splice) is slice:
            for i in range(index_or_splice.start, index_or_splice.stop, index_or_splice.step):
                remove_doc(i)
        else:
            remove_doc(index_or_splice)

    def __iter__(self):
        for i in range(self.next_doc_index):
            yield self.__getitem__(i)

    def clean_files(self):
        shutil.rmtree(self.temp_docs_dir)

    def __del__(self):
        CorpusDocMgr.__del__(self)
        self.clean_files()
