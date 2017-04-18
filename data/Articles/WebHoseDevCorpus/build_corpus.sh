NUM_ARTICLES=16275
dataset_dir_name="WebHoseDataset-${NUM_ARTICLES}-Articles"
rm -rf "$dataset_dir_name" >> /dev/null 2>&1
mkdir $dataset_dir_name
i=1
for article in `ls -1 /opt/nlp_shared/data/news_articles/webhose_english_dataset/ | shuf`; do
	cp "/opt/nlp_shared/data/news_articles/webhose_english_dataset/$article" "${dataset_dir_name}/${article}";
	((i++));
	if [[ "$i" -gt "${NUM_ARTICLES}" ]]; then
		break;
	fi
done
