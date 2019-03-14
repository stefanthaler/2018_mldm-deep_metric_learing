from __main__ import *

DATA_ROOT_DIR = "data"
DATASET_DIR = jp(DATA_ROOT_DIR,DATASET_NAME)

RAW_TRAIN_DATASET_FILE = jp(DATA_ROOT_DIR,DATASET_NAME, "raw_train.txt") # this is where the raw data should go
RAW_TRAIN_LABELS_FILE = jp(DATA_ROOT_DIR,DATASET_NAME, "raw_labels_train.txt") # this is where the raw data should go
RAW_TEST_DATASET_FILE = jp(DATA_ROOT_DIR,DATASET_NAME, "raw_test.txt")
RAW_TEST_LABELS_FILE = jp(DATA_ROOT_DIR,DATASET_NAME, "raw_labels_test.txt")

DATASET_PREPARATION_SCRIPT = jp(DATASET_DIR,"prepare_experiment_data.py")
DATASET_PREPARATION_MODULE = DATASET_PREPARATION_SCRIPT.replace(".py","").replace("/",".")

SEQUENCES_DIR = jp(DATASET_DIR, "sequences")
TRAIN_SEQUENCES_DIR = jp(DATASET_DIR, "sequences", "train")
TEST_SEQUENCES_DIR = jp(DATASET_DIR, "sequences", "test")

SEQUENCES_REVERSED_DIR = jp(DATASET_DIR, "sequences_reversed")


h.create_dir(DATA_ROOT_DIR)
h.create_dir(DATASET_DIR)

def check_dataset_preparation_script():
    import os
    if not os.path.exists(DATASET_PREPARATION_SCRIPT) or os.stat(DATASET_PREPARATION_SCRIPT).st_size<100:
        logger.warn("Expected to see %s, it does not exist. "%DATASET_PREPARATION_SCRIPT)
        with open(DATASET_PREPARATION_SCRIPT, "w+") as f:
            f.write("def download_and_prepare_dataset():\n")
            f.write("  assert False, 'Implement download_and_prepare_dataset method in %s'"%DATASET_PREPARATION_SCRIPT)
            f.write("  # 1 Download dataset \n")
            f.write("  # 2 Create vocabulary \n")
            f.write("  # 3 Prepare input for model\n")

        with open(jp(DATASET_DIR, "__init__.py"), "w+") as f:
            f.write("")
    else:
        logger.info("Dataset preparation script appears to be fine: %s "%DATASET_PREPARATION_SCRIPT)

check_dataset_preparation_script()
