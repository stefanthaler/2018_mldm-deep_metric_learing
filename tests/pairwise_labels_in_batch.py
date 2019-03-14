import tensorflow as tf
import numpy as np
import unittest
from unittest import *

# https://docs.python.org/3.5/library/unittest.html#assert-methods
# assertEqual(a, b) 	a == b 	 
# assertNotEqual(a, b) 	a != b 	 
# assertTrue(x) 	bool(x) is True 	 
# assertFalse(x) 	bool(x) is False 	 
# assertIs(a, b) 	a is b 	3.1
# assertIsNot(a, b) 	a is not b 	3.1
# assertIsNone(x) 	x is None 	3.1
# assertIsNotNone(x) 	x is not None 	3.1
# assertIn(a, b) 	a in b 	3.1
# assertNotIn(a, b) 	a not in b 	3.1
# assertIsInstance(a, b) 	isinstance(a, b) 	3.2
# assertNotIsInstance(a, b) 	not isinstance(a, b) 	3.2
def run_tests_on(pairwise_labels_in_batch):
    
    class TestPairwiseLabelsInBatch(TestCase):
        def setUp(self):
            self.labels = tf.placeholder(shape=[None], dtype=tf.int32)
            self.pw_lbl_eq_op = pairwise_labels_in_batch(self.labels)

            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())

        def test_all_in_batch(self):
            r = self.session.run(self.pw_lbl_eq_op, feed_dict={self.labels:[1,2,3,4]}).astype("int32")
            self.assertEqual(r.sum(), 16) # diagonoal should all be ones
            
        
        def test_none_in_batch(self):
            r = self.session.run(self.pw_lbl_eq_op, feed_dict={self.labels:[-1,-1,-1,-1]}).astype("int32")
            self.assertEqual(r.sum(), 0) # diagonoal should all be ones
            
        def test_some_in_batch(self):
            r = self.session.run(self.pw_lbl_eq_op, feed_dict={self.labels:[1,2,-1,-1]}).astype("int32")
            self.assertEqual(r.sum(), 4)
            self.assertTrue(np.array_equal(r, np.array([[1,1,0,0],[1,1,0,0],[0,0,0,0],[0,0,0,0]])))
        
        def test_some_in_batch2(self):
            r = self.session.run(self.pw_lbl_eq_op, feed_dict={self.labels:[1,-1,3,-1]}).astype("int32")
            self.assertEqual(r.sum(), 4)
            self.assertTrue(np.array_equal(r, np.array([[1,0,1,0],[0,0,0,0],[1,0,1,0],[0,0,0,0]])))
            

        def tearDown(self):
            self.session.close()
               
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromModule(TestPairwiseLabelsInBatch()))
    unittest.TextTestRunner().run(suite)  
    
    
    
    
    
    
