from __main__ import *

DATA_DIR = "data"
VIZUALIZATIONS_DIR = "visualizations"
INPUTS_DIR = jp(DATA_DIR, "inputs")
RESULTS_DIR = "results"

h.create_dir(DATA_DIR)  
h.create_dir(INPUTS_DIR)
h.create_dir(VIZUALIZATIONS_DIR) 
h.create_dir(RESULTS_DIR)
h.create_dir("graphs") 


TAG_NUM = -1 # set >1 to use a specific tag
if TAG_NUM < 0:
    TAG = "%0.3d"%(len(os.listdir("graphs"))+1)
    DO_TRAINING = True
else:
    TAG = "%0.3d"%(TAG_NUM)
    DO_TRAINING = False

GRAPH_DIR = jp("graphs", "%s-%s"%(MODEL_NAME, TAG))
h.create_dir(GRAPH_DIR) # store tensorflow calc graph here 

ENCODER_INPUTS_PATH = jp(DATA_DIR, "encoder_inputs", "%s.idx"%LOG_NAME)
ENC_SEQUENCE_LENGTH_PATH = jp(DATA_DIR, "sequence_lengths", "%s_enc.idx"%LOG_NAME)

SIGNATURE_FILE = jp(DATA_DIR, "signatures","%s.sig"%LOG_NAME)
SIGNATURES = np.array(list(open(SIGNATURE_FILE))).astype("int32")

SIGNATURES_BY_ID = {}
for sig_id, sig in enumerate(SIGNATURES):
    if not sig in SIGNATURES_BY_ID:
        SIGNATURES_BY_ID[sig]=[]
    SIGNATURES_BY_ID[sig].append(sig_id)