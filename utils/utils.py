import os
import errno
import torch
from torch.autograd import Variable
import copy
import numpy as np
import glob
import string
import pdb
import hashlib
import pickle

def check_table_exists(cur, table_name):
    cur.execute("select exists(select * from information_schema.tables where\
            table_name=%s)", (table_name,))
    return cur.fetchone()[0]

def save_object(file_name, data):
    with open(file_name, "wb") as f:
        res = f.write(pickle.dumps(data))

def load_object(file_name):
    res = None
    if os.path.exists(file_name):
        with open(file_name, "rb") as f:
            res = pickle.loads(f.read())
    return res

def deterministic_hash(string):
    return int(hashlib.sha1(str(string).encode("utf-8")).hexdigest(), 16)

def db_vacuum(conn, cur):
    old_isolation_level = conn.isolation_level
    conn.set_isolation_level(0)
    query = "VACUUM ANALYZE"
    cur.execute(query)
    conn.set_isolation_level(old_isolation_level)
    conn.commit()

def cosine_similarity_vec(vec1, vec2):
    cosine_similarity = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*
            np.linalg.norm(vec2))
    return cosine_similarity

def get_substr_words(words, substr):
    vals = []
    for w in words:
        if substr in w:
            vals.append(w)
    return vals

def get_regex_match_words(words, regex):
    vals = []
    for w in words:
        # if regex.match(w) is not None:
            # vals.append(w)
        if regex.search(w) is not None:
            vals.append(w)
    return vals

def get_match_vec(wv, pattern):
    '''
    @pattern is a string or a regex
    '''
    vals = []
    words = wv.index2word
    for w in words:
        if isinstance(pattern, str):
            if pattern in w:
                vals.append(wv[w])
        else:
            if pattern.search(w) is not None:
                vals.append(wv[w])

    return np.mean(np.array(vals), axis=0)

def clear_terminal_output():
    os.system('clear')

def preprocess_word(word, exclude_nums=False, exclude_the=False,
        exclude_words=[], min_len=0):
    word = str(word)
    # no punctuation
    exclude = set(string.punctuation)
    # exclude the as well
    if exclude_the:
        exclude.add("the")
    if exclude_nums:
        for i in range(10):
            exclude.add(str(i))

    # exclude.remove("%")
    word = ''.join(ch for ch in word if ch not in exclude)

    # make it lowercase
    word = word.lower()
    final_words = []

    for w in word.split():
        if w in exclude_words:
            continue
        if len(w) < min_len:
            continue
        final_words.append(w)

    return " ".join(final_words)

def to_variable(arr, use_cuda=True):
    if isinstance(arr, list) or isinstance(arr, tuple):
        arr = np.array(arr)
    if isinstance(arr, np.ndarray):
        arr = Variable(torch.from_numpy(arr), requires_grad=True)
    else:
        arr = Variable(arr, requires_grad=True)

    if torch.cuda.is_available() and use_cuda:
        arr = arr.cuda()
    return arr


def make_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def copy_network(Q):
    q2 = copy.deepcopy(Q)
    if torch.cuda.is_available():
        return q2.cuda()
    return q2

def save_network(model, name, step, out_dir, delete_old=False):
    '''
    saves the model for the given step, and deletes models for older
    steps.
    '''
    out_dir = '{}/models/'.format(out_dir)
    # Make Dir
    make_dir(out_dir)
    # find files in the directory that match same format:
    fnames = glob.glob(out_dir + name + "*")
    if delete_old:
        for f in fnames:
            # delete old ones
            os.remove(f)

    # Save model
    torch.save(model.state_dict(), '{}/{}_step_{}'.format(out_dir, name, step))

def model_name_to_step(name):
    return int(name.split("_")[-1])

def get_model_names(name, out_dir):
    '''
    returns sorted list of the saved model_step files.
    '''
    out_dir = '{}/models/'.format(out_dir)
    # Make Dir
    # find files in the directory that match same format:
    fnames = sorted(glob.glob(out_dir + name + "*"), key=model_name_to_step)
    return fnames

def get_model_name(args):
    if args.suffix == "":
        return str(hash(str(args)))
    else:
        return args.suffix

def adjust_learning_rate(args, optimizer, epoch):
    """
    FIXME: think about what makes sense for us?
    Sets the learning rate to the initial LR decayed by half every 30 epochs
    """
    # lr = args.lr * (0.1 ** (epoch // 30))
    lr = args.lr * (0.5 ** (epoch // 30))
    lr = max(lr, args.min_lr)
    if (epoch % 30 == 0):
        print("new lr is: ", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

####### Query Optimizer Utilities #########
def to_bitset(num_attrs, arr):
    ret = [i for i, val in enumerate(arr) if val == 1.0]
    for i, r in enumerate(ret):
        ret[i] = r % num_attrs
    return ret

def bitset_to_features(bitset, num_attrs):
    '''
    @bitset set of ints, which are the index of elements that should be in 1.
    Converts this into an array of size self.attr_count, with the appropriate
    elements 1.
    '''
    features = []
    for i in range(num_attrs):
        if i in bitset:
            features.append(1.00)
        else:
            features.append(0.00)
    return features

def check_actions_in_state(num_attrs, state, actions):
    sb = set(to_bitset(num_attrs, state))
    for a in actions:
        ab = set(to_bitset(num_attrs, a))
        if not ab.issubset(sb):
            return False
    return True

def find_cost(planOutput):
    '''
    parses planOutput to find the associated cost after the last join.
    '''
    all_lines = planOutput.split("\n")
    for s in all_lines:
        if "Join" in s:
            # the first Join we see would be the top most join.
            # JdbcJoin(condition=[=($40, $3)], joinType=[inner]): rowcount
            # = 480541.9921875, cumulative cost = {516195.625 rows, 1107.0
            words = s.split(" ")
            for i, w in enumerate(words):
                if w == "rows,":
                    cost = float(words[i-1].replace("{",""))
                    return cost

def get_berkeley_features(graph, num_attrs):
    '''
    @graph: is a parks/utils/DirectedGraph object
    @num_attrs: number of attributes in the dataset

    For each possible edge/action, we extract the berkeley features:
        state_vector (concat) left_vector (concat) right_vector

    @ret: (list, dict)
        list: each element will be a state+action feature vector.
        dict: action map. Each index of the list is the key that gives the
              appropriate edge / action.
    '''
    # FIXME: do this using graph.get_node_features_tensor
    # get the state set
    ret = []
    state_set = set()
    node_features, node_map = graph.get_node_features_tensor()
    for node_feature in node_features:
        state_set.update(node_feature)
    state_features = bitset_to_features(state_set, num_attrs)

    edge_features, edge_map = graph.get_edge_features_tensor()
    # for each edge, we featurize it and add the resulting vector to ret
    for i, edge_feature in enumerate(edge_features):
        edge = edge_map[i]
        # featurize left, and right side of the edge separately
        left = edge[0]
        # find all the visible attributes from the left side
        left_set = graph.get_node_feature(left)
        assert not edge_feature.isdisjoint(left_set)
        left_features = bitset_to_features(left_set, num_attrs)

        # all visible attributes from the right side
        right = edge[1]
        right_set = graph.get_node_feature(right)
        assert right_set != left_set
        assert not edge_feature.isdisjoint(right_set)
        right_features = bitset_to_features(right_set, num_attrs)
        # combine all three feature vectors and add to ret
        ret.append(state_features + left_features + right_features)
    return ret, edge_map
