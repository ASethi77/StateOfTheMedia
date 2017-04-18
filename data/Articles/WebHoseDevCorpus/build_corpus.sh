NUM_ARTICLES=100

for article in `ls -1 /opt/nlp_shared/data/news_articles/webhose_english_dataset/ | shuf`; do
	cp "/opt/nlp_shared/data/news_articles/webhose_english_dataset/$article" .;
	((i++));
	if [[ "$i" -gt "${NUM_ARTICLES}" ]]; then
		break;
	fi
done
