from nltk.tokenize import sent_tokenize, word_tokenize
import pickle
import os

class Document:
    DOC_PICKLE_POSTFIX = "-nlp-doc.dat"

    def load(doc_dir, name=""):
        doc_path = Document.get_filename(doc_dir, name)
        document = None
        try:
            with open(doc_path, "rb") as doc:
                document = pickle.load(doc)
        except FileNotFoundError as e:
            print("WARNING: Unable to load document {}. Returning empty placeholder instead".format(doc_path))
            return Document()

        return document

    def get_filename(doc_dir, doc_prefix):
        doc_name = doc_prefix + Document.DOC_PICKLE_POSTFIX
        doc_path = os.path.abspath(os.path.join(doc_dir, doc_name))
        return doc_path

    def __init__(self, content="", metadata={}, lang='en'):
        self.content = content
        self.metadata = metadata
        self.lang = lang

        self.sents = sent_tokenize(self.content)
        self.n_sents = len(self.sents)

    def __iter__(self):
        for token in word_tokenize(self.content):
            yield token

    def save(self, doc_dir, name=""):
        doc_path = Document.get_filename(doc_dir, name)
        with open(doc_path, "wb+") as doc:
            pickle.dump(self, doc)
