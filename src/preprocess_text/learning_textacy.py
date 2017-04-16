import textacy
gun_article = textacy.fileio.read.read_file("data/guns_r_bad.txt", mode=u'rt', encoding=None)
doc = textacy.Doc(gun_article)
print(doc)
bag_o_words = doc.to_bag_of_terms(normalize='lemma', as_strings=True, filter_stops=True)
print(bag_o_words)