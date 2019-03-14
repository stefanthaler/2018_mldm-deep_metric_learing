# https://www.python.org/dev/peps/pep-0257/
import nltk
import itertools
import time
import os
import library.helpers as h
from os.path import join as j
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_SEQ_LEN=2000

def default_tokenize(sentence, SPLIT_TOKEN=['.', '"' , "'" , ',' , '(', ')', '!', '?', ';', ':', "\\" , '/', '[',']',"=","-",'_' ]):
    # decode and to lower
    sentence = sentence.lower()
    if sentence[-1] == "\n":
        sentence = sentence[:-1]
    # add start and end token
    for char in SPLIT_TOKEN:
        sentence = sentence.replace(char, ' ' + char + ' ')
    # tokenize
    tokenized_logline = sentence.split(" ")[0:MAX_SEQ_LEN] #nltk.word_tokenize(logline)
    tokenized_logline = [t for t in tokenized_logline if len(t)>0]
    return tokenized_logline

PAD_TOKEN_ID = '0'
UNK_TOKEN_ID = '1'
START_TOKEN_ID = '2'
STOP_TOKEN_ID = '3'


class Vocabulary(object):

    def __init__(self):
        self.unique_tokens_and_counts={}
        self.token_to_index = {"PAD_TOKEN":PAD_TOKEN_ID, "UNK_TOKEN":UNK_TOKEN_ID, "START_TOKEN":START_TOKEN_ID, "STOP_TOKEN":STOP_TOKEN_ID}
        self.index_to_token = {PAD_TOKEN_ID:"PAD_TOKEN", UNK_TOKEN_ID:"UNK_TOKEN", START_TOKEN_ID:"START_TOKEN", STOP_TOKEN_ID:"STOP_TOKEN"}
        self.split_function = None
        self.log_name=None
        self.suffix=None
        self.max_seq_len=-1

    def _is_initialized(self):
        if not self.log_name is None or self.suffix is None or split_function is None:
            logger.warn("Vocabulary is probably not properly initialized")
            return False
        return True

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

    def save(self):
        i2t_path = j("data", "vocabularies", "%s%s_idx2tok.json"%(self.log_name, self.suffix))
        t2i_path = j("data", "vocabularies", "%s%s_tok2idx.json"%(self.log_name, self.suffix))
        voc_path = j("data", "vocabularies", "%s%s_voc.json"%(self.log_name, self.suffix))

        h.save_to_json(self.unique_tokens_and_counts, voc_path)
        h.save_to_json(self.token_to_index, t2i_path)
        h.save_to_json(self.index_to_token, i2t_path)
        logger.info("Done saving vocabulary '%s%s'"%(self.log_name, self.suffix))

    def words(self):
        return self.token_to_index.keys()

    @classmethod
    def exists(cls,log_name,suffix=""):
        i2t_path = j("data", "vocabularies", "%s%s_idx2tok.json"%(log_name, suffix))
        t2i_path = j("data", "vocabularies", "%s%s_tok2idx.json"%(log_name, suffix))
        voc_path = j("data", "vocabularies", "%s%s_voc.json"%(log_name, suffix))
        if not os.path.exists(i2t_path): return False
        if not os.path.exists(t2i_path): return False
        if not os.path.exists(voc_path): return False
        return True

    @classmethod
    def create(cls, log_name,suffix="", max_words_in_vocabulary =-1,  split_function=default_tokenize):
        num_special_token=4
        new_vocabulary = Vocabulary()
        new_vocabulary.log_name=log_name
        new_vocabulary.suffix=suffix
        new_vocabulary.split_function=split_function

        log_path = j("data", "raw", "%s%s.log"%(log_name,suffix))
        logger.info("Creating vocabulary for '%s'", log_path)
        # get word_frequencies for split lines
        word_frequencies = nltk.FreqDist(itertools.chain(*[split_function(line) for line in open(log_path, "r")]))
        logger.info("Counting word frequencies done.")
        logger.info("%i unique words in vocabulary"%(len(word_frequencies)))
        # determine vocabulary
        if max_words_in_vocabulary==-1:
            max_words = len(word_frequencies.items())
        else:
            max_words = max_words_in_vocabulary - num_special_token # because we add pad_token and unk_token
        new_vocabulary.unique_tokens_and_counts = {# merge two dicts
            **new_vocabulary.unique_tokens_and_counts,
            **dict(word_frequencies.most_common(max_words))
        }
        logger.info("Using %i words of vocabulary"%(new_vocabulary.size()))

        for i, word in enumerate(new_vocabulary.unique_tokens_and_counts.keys()):
            new_vocabulary.index_to_token["%s"%(i+num_special_token)]=word # +2 because we added PAD_TOKEN and UNK_TOKEN
            new_vocabulary.token_to_index[word]="%s"%(i+num_special_token)
        logger.info("Indexing done, vocabulary loaded.")
        return new_vocabulary

    @classmethod
    def load(cls, log_name, suffix="", split_function=default_tokenize):
        restored_vocabulary = Vocabulary()

        restored_vocabulary.log_name=log_name
        restored_vocabulary.suffix=suffix
        restored_vocabulary.split_function=split_function

        i2t_path = j("data", "vocabularies", "%s%s_idx2tok.json"%(log_name, suffix))
        t2i_path = j("data", "vocabularies", "%s%s_tok2idx.json"%(log_name, suffix))
        voc_path = j("data", "vocabularies", "%s%s_voc.json"%(log_name, suffix))


        # restore vocabulary
        logger.debug("Loading '%s'"%(voc_path))
        restored_vocabulary.unique_tokens_and_counts = {# merge two dicts
            **restored_vocabulary.unique_tokens_and_counts,
            **h.load_from_json(voc_path)
        }

        # restore index2tokenb
        logger.debug("Loading '%s'"%(i2t_path))
        restored_vocabulary.index_to_token = {# merge two dicts
            **restored_vocabulary.index_to_token,
            **h.load_from_json(i2t_path)
        }

        # restore vocabulary
        logger.debug("Loading '%s'"%(t2i_path))
        restored_vocabulary.token_to_index = {# merge two dicts
            **restored_vocabulary.token_to_index,
            **h.load_from_json(t2i_path)
        }

        return restored_vocabulary
