import os
import sys
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from importlib import import_module, reload


def run_tests_on(function_to_test):
    
    fn = function_to_test.__name__
    tests_fn = os.path.join("tests",fn+".py") 
    
    
    if not os.path.exists(tests_fn):
        logger.warn("Test file '%s' for function '%s' does not exist."%(tests_fn,fn,))
        logger.warn("Create this file and add a 'run_tests_on( <fucntion name> )' method in this file.")
        return False
    
    # dynamically get run_tests_method
    tests_module_name = "tests.%s"%fn
    if tests_module_name not in sys.modules:
        test_module = import_module(tests_module_name)
    else:
        reload(sys.modules[tests_module_name])
        test_module = sys.modules[tests_module_name]
    
    run_tests_method = getattr(test_module, "run_tests_on")

    # execute tests
    run_tests_method(function_to_test)
    