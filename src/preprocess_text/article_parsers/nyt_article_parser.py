import os
import glob
import json
from .article_parser import ArticleParser
from preprocess_text.document import Document

class NYTArticleParser(ArticleParser):
    def __init__(self, articles_directory):
        ArticleParser.__init__(self, articles_directory)
        self.json_article_files = glob.iglob(os.path.join(os.path.abspath(self.articles_dir), "*.json"))

    def num_articles(self):
        return len(glob.glob1(os.path.join(os.path.abspath(self.articles_dir)), "*.json"))

    def yield_articles(self):
        for article_filename in self.json_article_files:
            article_contents = ''
            with open(article_filename, "r") as article:
                article_contents = ' '.join(article.readlines())

            article_json = json.loads(article_contents.strip())
            if 'body' not in article_json:
                continue
            content = article_json["body"]
            del article_json["body"]
            metadata = article_json
            doc = Document(content, metadata=metadata, lang="en")
            doc.metadata["published"] = doc.metadata["publicationDate"]
            yield doc
