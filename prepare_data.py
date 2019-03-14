# generate vocabulary
from library.vocabulary import *
import library.helpers as h
import argparse
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
    Parse script arguments
"""
parser = argparse.ArgumentParser(description='adds a scheduled tasks')
parser.add_argument('-fn', '--file_name', type=str,default=None , help='Filename of the raw logs')
args = parser.parse_args()

file_name = args.file_name
logfile = open(j("data","raw","%s.log"%file_name))


"""
    Generate Vocabulary 
"""
vocab_exists = Vocabulary.exists(file_name,"")
if not vocab_exists:
    vocab = Vocabulary.create(file_name, "")
    vocab.save()
else:
    logger.info("\tVocabulary %s exists, skipping generation."%file_name)
    vocab = Vocabulary.load(file_name)

"""
    Generate Trainingsequences
"""
total_lines = h.num_lines(j("data","raw","%s.log"%file_name))

traingings_files = [
    j("data","encoder_inputs","%s.idx"%file_name),
    j("data","decoder_inputs","%s.idx"%file_name),
    j("data","decoder_targets","%s.idx"%file_name),
    j("data","sequence_lengths","%s_enc.idx"%file_name),
    j("data","sequence_lengths","%s_dec.idx"%file_name)]

all_files_are_there = True
for f in traingings_files:
    if not os.path.exists(f) or os.stat(f).st_size == 0:
        all_files_are_there=False
        break

if all_files_are_there:
    import sys
    logger.info("\t All inputs for '%s' exist, not regenerating them"%file_name)
    sys.exit(0)

encoder_input_seqs_file = open(j("data","encoder_inputs","%s.idx"%file_name),"w")
decoder_input_seqs_file = open(j("data","decoder_inputs","%s.idx"%file_name),"w")
decoder_target_seqs_file = open(j("data","decoder_targets","%s.idx"%file_name),"w")
enc_sequence_lengths_file = open(j("data","sequence_lengths","%s_enc.idx"%file_name),"w")
dec_sequence_lengths_file = open(j("data","sequence_lengths","%s_dec.idx"%file_name),"w")

for i,line in enumerate(logfile):
    if (i+1)%100==0:
        h.print_progress(i, total_lines, "Generating training sequences")

    # get sequences
    encoder_input_seq = list(vocab.line_to_index_seq(line))
    reversed_encoder_input_seq = list(encoder_input_seq) # copy
    reversed_encoder_input_seq.reverse() # reverse
    decoder_input_seq = [START_TOKEN_ID] + reversed_encoder_input_seq
    decoder_target_seq = reversed_encoder_input_seq + [STOP_TOKEN_ID]

    # write sequences
    encoder_input_seqs_file.write( " ".join(encoder_input_seq )+"\n" )
    enc_sequence_lengths_file.write("%s"%len(encoder_input_seq)+"\n")
    decoder_input_seqs_file.write( " ".join(decoder_input_seq )+"\n" )
    decoder_target_seqs_file.write(" ".join(decoder_target_seq)+"\n" )
    dec_sequence_lengths_file.write("%s"%len(decoder_input_seq)+"\n")

encoder_input_seqs_file.close()
decoder_input_seqs_file.close()
decoder_target_seqs_file.close()
enc_sequence_lengths_file.close()
dec_sequence_lengths_file.close()
