import os
import glob
import textacy
from .article_parser import ArticleParser
from preprocess_text.document import Document

class NYTArticleParser(ArticleParser):
    def __init__(self, articles_directory):
        ArticleParser.__init__(self, articles_directory)

    def yield_articles(self):
        json_article_files = glob.iglob(os.path.join(os.path.abspath(self.articles_dir), "*.json"))
        for article_filename in json_article_files:
            article_json = textacy.fileio.read.read_json(article_filename)
            if body not in article_json.keys():
                continue
            content, metadata = textacy.fileio.utils.split_record_fields(article_json, "body")
            content_full = ""
            for line in content:
                content_full += line.encode('ascii', errors='replace').decode('ascii')
            metadata_full = {}
            for data in metadata:
                metadata_full = {**metadata_full, **data}
            doc = Document(content_full, metadata=metadata_full, lang="en")
            yield doc
