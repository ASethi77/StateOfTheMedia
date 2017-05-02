import os
import glob
import textacy
from .article_parser import ArticleParser
from preprocess_text.document import Document

class WebhoseArticleParser(ArticleParser):
    def __init__(self, articles_directory):
        ArticleParser.__init__(self, articles_directory)
        self.json_article_files = glob.iglob(os.path.join(os.path.abspath(self.articles_dir), "news*.json"))

    def num_articles(self):
        return len(glob.glob1(os.path.join(os.path.abspath(self.articles_dir)), "news*.json"))

    def yield_articles(self):
        for article_filename in self.json_article_files:
            article_json = textacy.fileio.read.read_json(article_filename)
            content, metadata = textacy.fileio.utils.split_record_fields(article_json, "text")
            content_full = ""
            for line in content:
                content_full += line.encode('ascii', errors='replace').decode('ascii')
            metadata_full = {}
            for data in metadata:
                metadata_full = {**metadata_full, **data}
            yield Document(content_full, metadata=metadata_full, lang="en")
