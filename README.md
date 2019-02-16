# DB Embedding Tools

## Usage

To train the wordvec model given a db, and a set of queries in .sql files in
the given directory. Here is a sample execution, with arguments used by me in
my setup for a db named imdb, on default settings without any authentication.

```bash
$ python3 word2vec_embedding.py --db_host localhost --db_name imdb --sql_dir
new-queries/ --model_name new-wv-nopairs-10.bin --embedding_size 10 --data_dir
PATH/TO/STORE/MODEL/FILES/
```

There can be any number of sql files in `sql_dir` which will be executed as is,
and each generated row will be treated as a sentence in the learned model after
some common preprocessing.

To generate features from the given model, do:

```base
$ mkdir features
$ python3 gen_features.py --data_dir MODEL/LOCATION --model_name MODEL_YOU_TRAINED_BEFORE --embedding_len EMBEDDING_LEN_USED_FOR_MODEL --add_count
```

## Notes:

* For other flags, the default values should all be fine just now. Note: when running
on the same data twice, this tries to read it from the disk - so if you
cancelled a previous run early, then use flag --regen_sentences.

* The normal use case is to load everything into memory and then run the
wordvec training. This is much faster than using a streaming model from the db
(--sentence_gen), or using a file from the disk (--no_pickle). But for
processing really large number of rows (50m+ on my setup), I occasionally ran
into memory issues with loading everything in.

