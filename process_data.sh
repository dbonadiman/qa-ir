python -m qair.data.reader wikiqa WikiQA-train.tsv WikiQA-dev.tsv WikiQA-test.tsv
python -m qair.data.reader trecqa train2393.cleanup.xml dev-less-than-40.manual-edit.xml test-less-than-40.manual-edit.xml
python -m qair.data.tokenizer wikiqa
python -m qair.data.tokenizer trecqa
python -m qair.data.vocabulary wikiqa --lower
python -m qair.data.vocabulary trecqa --lower
python -m qair.data.embeddings wikiqa alexi.bin 
python -m qair.data.embeddings trecqa alexi.bin 
python -m qair.data.embeddings wikiqa glove.txt 
python -m qair.data.embeddings trecqa glove.txt 