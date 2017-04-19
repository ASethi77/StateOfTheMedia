import textacy

# articles = textacy.fileio.read_json_lines("../data/Articles/WebHoseDevCorpus/merged_ascii.json")
# texts, metadatas = textacy.fileio.split_record_fields(articles, content_field='text')
#
# split_texts = list(texts)
# split_metadatas = list(metadatas)
# bad_article, bad_metadata = (split_texts[1], split_metadatas[1])

# article_corpus = textacy.Corpus('en', texts=texts, metadatas=metadatas)
# article_corpus.save(path='../data/TextacyCorpra/', name='WebHoseDevCorpus')


# doc = textacy.Doc(content=bad_article.encode('ascii', errors='replace').decode('ascii'), lang='en')
# doc.save("../data/TextacyCorpra", name="dumb")

# doc = textacy.Doc("ŷ", lang="en")
# doc = textacy.Doc.load("../../data/TextacyCorpra", name="dumb")
# print(doc)

corp = textacy.Corpus.load(path='../data/TextacyCorpra/', name='WebHoseDevCorpus')
print(corp)

# with open("../data/Articles/WebHoseDevCorpus/merged_ascii.json", mode="w+") as f:
#     for line in textacy.fileio.read_file_lines("../data/Articles/WebHoseDevCorpus/merged.json"):
#         ascii_line = line.encode('ascii', errors='replace').decode('ascii')
#         f.write(ascii_line)

