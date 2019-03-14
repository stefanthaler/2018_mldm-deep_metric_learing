from __main__ import *

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

# fix training
RANDOM_SEED = 0 
# configure numpy 
np.set_printoptions(precision=3, suppress=True)
np.random.seed(RANDOM_SEED)

# configure tensorflow
tf.set_random_seed(RANDOM_SEED)

# configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# configure ipython display
def show(img_file):
    try: # only works in ipython notebook
        display(Image(filename=img_file))
    except:
        pass