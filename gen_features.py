import sqlparse
import glob
import pdb
from collections import defaultdict
import argparse
from utils.utils import *
from gensim.models import Word2Vec
import time
import re
import pickle

CMP_OP_ID = 0

def get_next_cmp_op_id():
    global CMP_OP_ID
    CMP_OP_ID += 1
    return CMP_OP_ID-1

def get_unknown_feature_vec():
    vec = []
    for i in range(args.embedding_len):
        vec.append(0.00)
    return np.array(vec)

def handle_token_list(token_list, data, identifier="", cmp_op=""):
    in_predicate = True
    index = 0
    if identifier == "":
        in_predicate = False
    while (True):
        index, token = token_list.token_next(index)
        if args.debug: print("next token: ", token)
        if token is None:
            break
        # now we can deal differently with different types of tokens
        if args.debug: print(token.ttype, token)
        if (type(token) == sqlparse.sql.Comparison):
            # comparison seems to have the value within it as well
            # token.left / token.right should give us what we need
            identifier = token.left.value
            cmp_op = token.token_next(0)[1].value
            # FIXME: add comparison operator as well
            data[identifier] = (cmp_op, [token.right.value])
            identifier = ""
            in_predicate = False
            cmp_op = ""
        elif (type(token) == sqlparse.sql.Identifier):
            identifier = token.value
            in_predicate = True
        else:
            # parenthesis can start ANYWHERE
            if args.debug: print(type(token), str(token.ttype))
            if (type(token) == sqlparse.sql.Parenthesis):
                # handle the stuff inside parenthesis separately
                handle_token_list(token, data, identifier, cmp_op)
                # this should always be true at the end
                in_predicate = False
                cmp_op = ""
                identifier = ""
                continue
            elif ("Punctuation" in str(token.ttype)):
                continue
            elif not in_predicate:
                # just valid for JOB
                assert (token.value == "AND" or token.value == "OR")
                continue
            elif (type(token) == sqlparse.sql.IdentifierList):
                # FIXME: deal with the identifier list!
                vals = token.get_identifiers()
                in_predicate = False
                vals = [v.value for v in vals]
                data[identifier] = (cmp_op, vals)
                cmp_op = ""
                identifier = ""
                continue
            elif token.is_keyword:
                cmp_op += " " + token.value
            elif ("Literal" in str(token.ttype)):
                if args.debug: print(token.ttype, token)
                # deal with it.
                in_predicate = False
                values = []
                values.append(token.value)
                if ("BETWEEN" in cmp_op):
                    # this should be AND
                    index, token = token_list.token_next(index)
                    assert str(token.value) == "AND"
                    index, token = token_list.token_next(index)
                    # FIXME: handle this
                    values.append(token.value)

                data[identifier] = (cmp_op, values)
                cmp_op = ""
                identifier = ""
                continue

def get_feature_vec(vals):
    feature_vec = []
    cmp_op_vec = []
    for i in range(CMP_OP_ID):
        cmp_op_vec.append(0.00)

    for cmp_op_id in vals[0]:
        assert cmp_op_id < CMP_OP_ID
        cmp_op_vec[cmp_op_id] = 1.00

    num_vals = [vals[1]]
    wv = vals[2]
    wv = list(wv)

    if args.no_wv:
        return cmp_op_vec + [wv[-1]]
    else:
        return cmp_op_vec + num_vals + wv

def write_out_features(final_vectors):
    print("final pass to write out feature vectors")
    for fname, data in final_vectors.items():
        # find new file name
        out_name = os.path.basename(fname)
        out_name = out_name.replace(".sql", ".pickle")
        out_name = args.features_dir + "/" + out_name
        f = open(out_name, "wb")
        out_dict = {}
        for attr, vals in data.items():
            if args.add_count:
                assert len(vals[2]) == args.embedding_len+1
            else:
                assert len(vals[2]) == args.embedding_len
            feature_vec = get_feature_vec(vals)
            out_dict[attr] = feature_vec
        f.write(pickle.dumps(out_dict))
        f.close()

def main():
    file_names = glob.glob("job/*.sql")
    queries = []
    extracted_data = {}
    # match each cmp op, with it's index in 1-hot vector
    cmp_ops = {}
    # final vectors: query : attribute : feature vector
    #   feature vector is: [CMP OP ONE HOT] [NUM ATTRS] [WORD VECTOR]
    #   WORD VECTOR can either be the sum of a bunch of word vectors or the
    #   mean (OR v/s AND)
    final_vectors = {}

    for fn in file_names:
        with open(fn, "rb") as f:
            queries.append(f.read())
        extracted_data[fn] = defaultdict(tuple)

    # what we want:
    #   query : table_name : attribute name : ([...comparison types...], [feature
    #   vectors])
    # final feature vector description:
    #   - [comparison_ops] + [scalar: num_feature_vectors] + mean([all feature
    #   vectors])
    # can average out all the feature vectors etc.
    # AND / OR etc. not easy to distinguish right now...

    for i, q in enumerate(queries):
        if args.debug: print(sqlparse.format(q, reindent=True))
        parsed = sqlparse.parse(q)[0]
        # let us go over all the where clauses
        where_clauses = None
        for token in parsed.tokens:
            if (type(token) == sqlparse.sql.Where):
                where_clauses = token

        token_list = sqlparse.sql.TokenList(where_clauses)
        file_name = file_names[i]
        handle_token_list(where_clauses, extracted_data[file_name])

    # load model
    # model_dir = args.data_dir
    # model_name = model_dir + "all_attributes.bin"
    # model_name = model_dir + "all_attributes_split_words.bin"
    # model_name = model_dir + "preprocessed-words-model.bin"
    # model_name = model_dir + "joined-tables-half.bin"
    # model_name = model_dir + "all-w2v-nopairs25.bin"
    # model_name = model_dir + "new-wv-nopairs25.bin"
    model_name = args.data_dir + args.model_name

    model = Word2Vec.load(model_name)
    print(model)
    wv = model.wv
    del model
    total_found = 0
    total_not_found = 0
    all_not_found = []
    total_like = 0

    for query, data in extracted_data.items():
        print(query)
        found = 0
        not_found = 0
        like = 0
        final_vectors[query] = {}
        for attribute,vals in data.items():
            cmp_op = vals[0]
            # FIXME: deal with these when extracting etc.
            cmp_op = cmp_op.replace("NOT NULL", "")
            cmp_op = cmp_op.replace("NULL", "")
            cmp_op = cmp_op.replace("IS", "")
            cmp_op = cmp_op.replace("AND", "")
            cmp_op = cmp_op.replace(" ", "")

            if cmp_op not in cmp_ops:
                next_cmp_id = get_next_cmp_op_id()
                cmp_ops[cmp_op] = next_cmp_id

            literal_vals = []
            if "not null" in cmp_op or "NOT NULL" in cmp_op:
                # FIXME:
                continue
            elif "LIKE" in cmp_op or "like" in cmp_op:
                # FIXME: ( ...), $ signs
                like += 1
                like_val = vals[1]
                like_val = preprocess_word(like_val)
                # python regex
                like_val = like_val.replace("%", ".*")
                like_val = like_val.replace("'", "")
                like_regex = re.compile(like_val)
                literal_vals = get_regex_match_words(wv.index2word, like_regex)
            else:
                literal_vals = vals[1]

            # add all these values so far?
            # TODO: add feature indicating how many literal_vals were there.
            # there can be many values because of IN (....), or LIKE regex
            # matches.
            num_matches = 0
            matched_vectors = []
            total_count = 0
            for val in literal_vals:
                # FIXME: not doing anything for join conditions
                if ("." in val):
                    continue
                num_matches += 1
                preprocessed_val = preprocess_word(val)
                if preprocessed_val in wv:
                    found += 1
                    matched_vectors.append(wv[preprocessed_val])
                    total_count += wv.vocab[preprocessed_val].count
                elif len(preprocessed_val.split()) > 1:
                    # separate out each value into individual words too
                    word_vectors = []
                    for word in preprocessed_val.split():
                        if word not in wv:
                            not_found += 1
                            all_not_found.append(word)
                            word_vectors.append(get_unknown_feature_vec())
                            continue
                        word_vectors.append(wv[word])
                        total_count += wv.vocab[word].count
                    word_vectors = np.array(word_vectors)
                    matched_vectors.append(np.mean(word_vectors, axis=0))
                else:
                    print("not found")
                    print("attribute: ", attribute)
                    print("cmp op: ", cmp_op)
                    print("orig: ", val)
                    print("preprocessed: ", preprocessed_val)
                    not_found += 1
                    all_not_found.append(val)
                    matched_vectors.append(get_unknown_feature_vec())

            if len(matched_vectors) != num_matches:
                pdb.set_trace()
            if len(matched_vectors) == 0:
                continue
            matched_vectors = np.array(matched_vectors)

            # FIXME: what is the best way to deal with this? I think always
            # choose mean seems more sensible.
            # final_wv = np.sum(matched_vectors, axis=0)
            final_wv = np.mean(matched_vectors, axis=0)
            if args.add_count:
                final_wv = np.append(final_wv, total_count)
                assert len(final_wv) == args.embedding_len+1

            if attribute in final_vectors[query]:
                pdb.set_trace()
                # then take mean of this vector, and what exists there. + add
                # new cmp
                old_result = final_vectors[query][attribute]
                all_wv = [old_result[2], final_wv]
                new_wv = np.mean(np.array(all_wv), axis=0)
                new_result = ([old_result[0][0], cmp_ops[cmp_op]],
                        len(matched_vectors)+old_result[1], new_wv)
                final_vectors[query][attribute] = new_result
            else:
                result = ([cmp_ops[cmp_op]], len(matched_vectors), final_wv)
                final_vectors[query][attribute] = result

        print("found: {}, not_found: {}, like: {}".format(found, not_found,
            like))
        total_found += found
        total_not_found += not_found
        total_like += like

    assert total_not_found == len(all_not_found)
    for w in all_not_found:
        print(w)
    print("found: {}, not_found: {}, like: {}".format(total_found, total_not_found,
        total_like))
    regex = re.compile(".*robert.*")
    matches = get_regex_match_words(wv.index2word, regex)

    # now can use final vectors to write out all the files
    write_out_features(final_vectors)

def read_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=False,
            default="/data/pari/embeddings/word2vec/")
    parser.add_argument("--model_name", type=str, required=False,
            default="test.bin")
    parser.add_argument("--features_dir", type=str, required=False,
            default="./features/")
    parser.add_argument("--embedding_len", type=int, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--add_count", action="store_true")
    parser.add_argument("--no_wv", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = read_flags()
    main()
