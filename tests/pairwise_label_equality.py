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
def run_tests_on(pairwise_label_equality):
    
    class TestPairwiseLabelEquality(TestCase):
        def setUp(self):
            self.labels = tf.placeholder(shape=[None], dtype=tf.int32)
            self.pw_lbl_eq_op = pairwise_label_equality(self.labels)

            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())

        def test_all_different(self):
            r = self.session.run(self.pw_lbl_eq_op, feed_dict={self.labels:[1,2,3,4]}).astype("int32")
            self.assertEqual(r.sum(), 4) # diagonoal should all be ones
            self.assertTrue(np.array_equal(np.eye(4),r) ) # diagonoal should all be ones
        
        def test_all_same(self):
            r = self.session.run(self.pw_lbl_eq_op, feed_dict={self.labels:[-1,-1,-1,-1]}).astype("int32")
            self.assertEqual(r.sum(), 16) # diagonoal should all be ones
            self.assertTrue(np.array_equal(np.ones(shape=(4,4)) ,r) ) # diagonoal should all be ones
            
        def test_few_different(self):
            r = self.session.run(self.pw_lbl_eq_op, feed_dict={self.labels:[1,2,1,2]}).astype("int32")
            self.assertEqual(r.sum(), 8)
            self.assertEqual(r[0,0], 1)
            self.assertEqual(r[1,1], 1)
            self.assertEqual(r[2,2], 1)
            self.assertEqual(r[3,3], 1)
            self.assertEqual(r[0,2], 1)
            self.assertEqual(r[1,3], 1)
            self.assertEqual(r[2,0], 1)
            self.assertEqual(r[3,1], 1)
            

        def tearDown(self):
            self.session.close()
               
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromModule(TestPairwiseLabelEquality()))
    unittest.TextTestRunner().run(suite)  
    
    
    
    
    
    
