import os
import glob
import json
import textacy
from article_parser import ArticleParser

class WebhoseArticleParser(ArticleParser):
	def __init__(self, articles_directory):
		ArticleParser.__init__(self, articles_directory)

	def yield_articles(self):
		json_article_files = glob.iglob(os.path.join(os.path.abspath(self.articles_dir), "news*.json"))
		for article_filename in json_article_files:
			article_json = textacy.fileio.read.read_json(article_filename)
			content, metadata = textacy.fileio.utils.split_record_fields(article_json, "text")
			content_full = ""
			for line in content:
				content_full += line
			yield textacy.Doc(content_full, metadata=metadata, lang="en")
