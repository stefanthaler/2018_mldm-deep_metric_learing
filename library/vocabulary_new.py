from __main__ import *

def default_tokenize(sentence, SPLIT_TOKEN=['.', '"' , "'" , ',' , '(', ')', '!', '?', ';', ':', "\\" , '/', '[',']',"=","-",'_' ]):
    # decode and to lower
    sentence = sentence.lower()
    if sentence[-1] == "\n":
        sentence = sentence[:-1]
    # add start and end token
    for char in SPLIT_TOKEN:
        sentence = sentence.replace(char, ' ' + char + ' ')
    # tokenize
    tokenized_logline = sentence.split(" ")
    tokenized_logline = [t for t in tokenized_logline if len(t)>0]
    return tokenized_logline

PAD_TOKEN_ID = '0'
UNK_TOKEN_ID = '1'
START_TOKEN_ID = '2'
STOP_TOKEN_ID = '3'

"""
    Vocabulary expects the following file to be present:

    data/{data_set_name}/raw.txt
"""
class Vocabulary(object):

    def __init__(self, data_set_name, is_training_dataset=True, force_regeneration=False, split_function=default_tokenize, max_seq_len=-1, max_tokens=-1):
        self.unique_tokens_and_counts={}
        self.token_to_index = {"PAD_TOKEN":PAD_TOKEN_ID, "UNK_TOKEN":UNK_TOKEN_ID, "START_TOKEN":START_TOKEN_ID, "STOP_TOKEN":STOP_TOKEN_ID}
        self.index_to_token = {PAD_TOKEN_ID:"PAD_TOKEN", UNK_TOKEN_ID:"UNK_TOKEN", START_TOKEN_ID:"START_TOKEN", STOP_TOKEN_ID:"STOP_TOKEN"}
        self.num_special_token = len(self.token_to_index)
        self.split_function = split_function # how to split a line into tokens
        self.data_set_name=data_set_name # the name of our dataset
        self.max_tokens=max_tokens  # number of tokens in the vocabulary
        self.max_seq_len=max_seq_len # maximum number of split tokens that one line is allowed to have

        if is_training_dataset:
            prefix = "train"
            self.vocabulary_input_file = RAW_TRAIN_DATASET_FILE
        else:
            prefix="test"
            self.vocabulary_input_file = RAW_TEST_DATASET_FILE

        self.i2t_path = jp(DATA_ROOT_DIR, self.data_set_name, "vocabulary", "%s_idx2tok.json"%prefix)
        self.t2i_path = jp(DATA_ROOT_DIR, self.data_set_name, "vocabulary", "%s_tok2idx.json"%prefix)
        self.voc_path = jp(DATA_ROOT_DIR, self.data_set_name, "vocabulary", "%s_tokens_and_counts.json"%prefix)

        h.create_dir( jp(DATA_ROOT_DIR, self.data_set_name, "vocabulary") )

        if self.exists() and not force_regeneration:
            # load vocabulary
            self.__load()
        else:
            self.__initialize()
            self.__save()

    def token_to_idx(self, token): # token_to_idx("hello") => 12
        if token in self.token_to_index:
            return self.token_to_index[token]
        else:
            return UNK_TOKEN_ID

    def idx_to_token(self, idx): # token_to_idx(12) => "hello"
        idx = "%s"%idx
        if idx in self.index_to_token:
            return self.index_to_token[idx]
        else:
            return "UNK_TOKEN"

    def size(self):
        assert(len(self.token_to_index)==len(self.index_to_token))
        return len(self.index_to_token)

    def line_to_index_seq(self, line): #"the cat sat" => [14,12,15]
        return map(self.token_to_idx, self.split_function(line))

    def index_seq_to_line(self, index_seq): #[14,12,15] => "the cat sat"
        return " ".join(map(self.idx_to_token, index_seq ))

    def words(self):
        return self.token_to_index.keys()

    def exists(self):
        if not os.path.exists(self.i2t_path): return False
        if not os.path.exists(self.t2i_path): return False
        if not os.path.exists(self.voc_path): return False
        return True

    def delete(self): # deletes vocabulary files from disk if they exist
        for f in [self.i2t_path, self.t2i_path, self.voc_path]:
            if os.path.exists(f):
                os.remove(f)
                logger.info("Deleted %s"%f)

    def __save(self):
        h.save_to_json(self.unique_tokens_and_counts, self.voc_path)
        h.save_to_json(self.token_to_index, self.t2i_path)
        h.save_to_json(self.index_to_token, self.i2t_path)
        logger.info("Done saving vocabulary '%s'"%(self.data_set_name))

    def __initialize(self):
        if not os.path.exists(self.vocabulary_input_file):
            logger.warn("Expexted raw text file '%s' to exists."%self.vocabulary_input_file)
            assert False, "Expected %s to exists"%self.vocabulary_input_file

        logger.info("Creating vocabulary for '%s'", self.vocabulary_input_file)

        # get word_frequencies for split lines
        word_frequencies = nltk.FreqDist(itertools.chain(*[self.split_function(line) for line in open(self.vocabulary_input_file, "r")]))
        logger.info("Counting word frequencies done.")
        logger.info("%i unique words in vocabulary"%(len(word_frequencies)))
        # determine vocabulary
        if self.max_tokens==-1:
            max_words = len(word_frequencies.items())
            logger.info("Using all words and ") #TODO continue here;
        else:
            max_words = self.max_tokens - self.num_special_token # because we add pad_token and unk_token

        self.unique_tokens_and_counts = {# merge two dicts
            **self.unique_tokens_and_counts,
            **dict(word_frequencies.most_common(max_words))
        }
        logger.info("Using %i words of vocabulary"%(self.size()))

        for i, word in enumerate(self.unique_tokens_and_counts.keys()):
            self.index_to_token["%s"%(i+self.num_special_token)]=word # +2 because we added PAD_TOKEN and UNK_TOKEN
            self.token_to_index[word]="%s"%(i+self.num_special_token)
        logger.info("Indexing done, vocabulary loaded.")

    def __load(self):
        if not self.exists():
            logger.warn("Cannot load vocabulary '%s', some or all of the vocabulary files are missing."%self.data_set_name)
            return

        # restore vocabulary
        logger.debug("Loading '%s'"%(self.voc_path))
        self.unique_tokens_and_counts = {# merge two dicts
            **self.unique_tokens_and_counts,
            **h.load_from_json(self.voc_path)
        }

        # restore index2token
        logger.debug("Loading '%s'"%(self.i2t_path))
        self.index_to_token = {# merge two dicts
            **self.index_to_token,
            **h.load_from_json(self.i2t_path)
        }

        # restore vocabulary
        logger.debug("Loading '%s'"%(self.t2i_path))
        self.token_to_index = {# merge two dicts
            **self.token_to_index,
            **h.load_from_json(self.t2i_path)
        }
