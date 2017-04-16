import os

class ArticleParser:
	def __init__(self, articles_directory):
		if (os.path.isdir(articles_directory)):
			self.articles_dir = articles_directory
		else:
			raise ValueError("Path to articles directory is invalid.")
		
	def yield_articles(self):
		raise NotImplementedError
