mkdir data
mkdir data/raw
cd original_data
wget http://cs.jhu.edu/~xuchen/packages/jacana-qa-naacl2013-data-results.tar.bz2
tar xvjf jacana-qa-naacl2013-data-results.tar.bz2
unzip WikiQACorpus.zip
cd ..
mv tmp/jacana-qa-naacl2013-data-results data/raw/trecqa
mv tmp/WikiQACorpus data/raw/wikiqa

gzip -d data/raw/trecqa/train2393.cleanup.xml.gz