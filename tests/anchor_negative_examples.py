import tensorflow as tf
import numpy as np
import unittest
from unittest import *

# pw_label_equality
pw_le_all_equal = np.array([[1,1,1],[1,1,1],[1,1,1]])
pw_le_some_equal = np.array([[1,1,0],[1,1,0],[0,0,1]])
pw_le_non_equal = np.array([[1,0,0],[0,1,0],[0,0,1]])

# pw_label_in_batch
pw_lib_all_in = np.array([[1,1,1],[1,1,1],[1,1,1]])
pw_lib_non_in = np.array([[0,0,0],[0,0,0],[0,0,0]])
pw_lib_some_in = np.array([[0,0,1],[0,0,1],[1,1,1]])

# pw_jaccard_distances
pw_jds = np.array([[0,0.2,0.8],[0.2,0,0.5],[0.8,0.5,0]])

# pw_euclidean_distances
pw_eds = np.array([[0,0.1,0.7],[0.1,0,0.6],[0.7,0.6,0]])
# [[-1,0.1,0.7],[0.1,-1,0.6],[0.7,0.6,-1]]


def run_tests_on(anchor_negative_examples):
    
    class TestAnchorNegativeExamples(TestCase):
        def setUp(self):
            self.pw_label_equality=tf.placeholder(shape=[None, None], dtype=tf.int32) # [batch_size, batch_size] 
            self.labels_in_batch=tf.placeholder(shape=[None, None], dtype=tf.bool) # [batch_size, batch_size] 
            self.pw_jaccard_distances= tf.placeholder(shape=[None, None], dtype=tf.float32) # [batch_size, batch_size] 
            self.pw_euclidean_distances= tf.placeholder(shape=[None, None], dtype=tf.float32) # [batch_size, batch_size] 
            self.jd_pos_threshold=tf.placeholder(shape=(), dtype=tf.float32)
            
            self.op = anchor_negative_examples(self.pw_label_equality, self.labels_in_batch, self.pw_jaccard_distances, self.pw_euclidean_distances, self.jd_pos_threshold )

            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())
            
        def run_op(self, le, lib, jds, eds, jd_t):#shortcut
            return self.session.run(self.op,feed_dict={
                self.pw_label_equality:le, 
                self.labels_in_batch:lib, 
                self.pw_jaccard_distances:jds, 
                self.pw_euclidean_distances:eds, 
                self.jd_pos_threshold:jd_t
            })
        
        def ex_sum(self, op ):
            return self.session.run(tf.reduce_sum(tf.cast(op,tf.int32)))
        
        # all examples are labeled, all equal
        def test_all_in_all_equal_all_jd(self):
            r_ed, r_bol , r_boj = self.run_op(pw_le_all_equal, pw_lib_all_in, pw_jds, pw_eds, 0.0)
            self.assertEqual(0, self.ex_sum(r_bol) ) # all examples labeled, but there are no positive ones
            self.assertEqual(0, self.ex_sum(r_boj) ) # because all examples are labelled, ignore jaccard distance        
            self.assertTrue(np.array_equal(np.array([[-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0]], dtype="float32"),r_ed) ) # all euclidean distances should be passed through
        
        def test_all_in_all_equal_no_jd(self):
            r_ed, r_bol , r_boj = self.run_op(pw_le_all_equal, pw_lib_all_in, pw_jds, pw_eds, 1.0)
            self.assertEqual(0, self.ex_sum(r_bol) ) # all examples labeled, exclude pos to itself
            self.assertEqual(0, self.ex_sum(r_boj) ) # because all examples are labelled, ignore                   
            self.assertTrue(np.array_equal(np.array([[-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0]], dtype="float32"),r_ed) ) # all euclidean distances should be passed through
        
        def test_all_in_some_equal_all_jd(self):
            r_ed, r_bol , r_boj = self.run_op(pw_le_some_equal, pw_lib_all_in, pw_jds, pw_eds, 0.0)
            self.assertEqual(4, self.ex_sum(r_bol) ) # all examples labeled, but there are no positive ones
            self.assertEqual(0, self.ex_sum(r_boj) ) # because all examples are labelled, ignore jaccard distance        
            self.assertTrue(np.array_equal(np.array([[-1.0,-1.0,0.7],[-1.0,-1.0,0.6],[0.7,0.6,-1.0]], dtype="float32"),r_ed) ) # all euclidean distances should be passed through
        
        def test_all_in_some_equal_no_jd(self):
            r_ed, r_bol , r_boj = self.run_op(pw_le_some_equal, pw_lib_all_in, pw_jds, pw_eds, 1.0)
            self.assertEqual(4, self.ex_sum(r_bol) ) # all examples labeled, exclude pos to itself
            self.assertEqual(0, self.ex_sum(r_boj) ) # because all examples are labelled, ignore                   
            self.assertTrue(np.array_equal(np.array([[-1.0,-1.0,0.7],[-1.0,-1.0,0.6],[0.7,0.6,-1.0]], dtype="float32"),r_ed) ) # all euclidean distances should be passed through
        
        
        # no examples are labeled
        def test_non_in_all_jd_all_la_eq(self):
            r_ed, r_bol , r_boj = self.run_op(pw_le_all_equal, pw_lib_non_in, pw_jds, pw_eds, 0.0)
            self.assertEqual(0, self.ex_sum(r_bol) ) # ignore, because no examples are labeled 
            self.assertEqual(6, self.ex_sum(r_boj) ) # because all pw jd are greater than 0.0
            self.assertTrue(np.array_equal(np.array([[-1,0.1,0.7],[0.1,-1,0.6],[0.7,0.6,-1]], dtype="float32"),r_ed) ) # all euclidean distances should be passed through
        
        def test_non_in_two_jd_all_la_eq(self):
            r_ed, r_bol , r_boj = self.run_op(pw_le_all_equal, pw_lib_non_in, pw_jds, pw_eds, 0.5)
            self.assertEqual(0, self.ex_sum(r_bol) ) # ignore, because no examples are labeled 
            self.assertEqual(4, self.ex_sum(r_boj) ) # only two pw jd greater than or equal 0.5 
            self.assertTrue(np.array_equal(np.array([[-1,-1.0,0.7],[-1.0,-1,0.6],[0.7,0.6,-1]], dtype="float32"),r_ed) ) # all euclidean distances should be passed through
            
        def test_non_in_all_jd_no_la_eq(self):
            r_ed, r_bol , r_boj = self.run_op(pw_le_non_equal, pw_lib_non_in, pw_jds, pw_eds, 0.0)
            self.assertEqual(0, self.ex_sum(r_bol) ) # ignore, because no examples are labeled 
            self.assertEqual(6, self.ex_sum(r_boj) ) # because not all jd are gteq 0.5             
            self.assertTrue(np.array_equal(np.array([[-1,0.1,0.7],[0.1,-1,0.6],[0.7,0.6,-1]], dtype="float32"),r_ed) ) # all euclidean distances should be passed through
        
        def test_non_in_two_jd_no_eq(self):
            r_ed, r_bol , r_boj = self.run_op(pw_le_non_equal, pw_lib_non_in, pw_jds, pw_eds, 0.5)
            self.assertEqual(0, self.ex_sum(r_bol) ) # ignore, because no examples are labeled 
            self.assertEqual(4, self.ex_sum(r_boj) ) # because all examples have a jd > 0                  
            self.assertTrue(np.array_equal(np.array([[-1,-1.0,0.7],[-1.0,-1,0.6],[0.7,0.6,-1]], dtype="float32"),r_ed) ) # all euclidean distances should be passed through
        
        # some examples labeled, others not 
        def test_some_in_all_lb_eq_no_jd(self):
            r_ed, r_bol , r_boj = self.run_op(pw_le_all_equal, pw_lib_some_in, pw_jds,  pw_eds, 0.0)
            self.assertEqual(0, self.ex_sum(r_bol) ) # all labels are equal, so non is negative because of that
            self.assertEqual(2, self.ex_sum(r_boj) ) # all jds are larger than 0, but all labeled examples are equal 
            self.assertTrue(np.array_equal(np.array([[-1.0,0.1,-1.0],[0.1,-1.0,-1],[-1,-1,-1.0]], dtype="float32"),r_ed) ) # all euclidean distances should be passed through
        
        def test_some_in_all_lb_eq_jd_greater(self):
            r_ed, r_bol , r_boj = self.run_op(pw_le_all_equal, pw_lib_some_in,pw_jds,  pw_eds, 0.6)
            self.assertEqual(0, self.ex_sum(r_bol) ) # all labels are equal, so non is negative because of that
            self.assertEqual(0, self.ex_sum(r_boj) ) # only 0.8 is greater, but it is ignored because we have labels that show otherwise                   
            self.assertTrue(np.array_equal(np.array([[-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0]], dtype="float32"),r_ed) ) # all euclidean distances should be passed through
            
        def test_some_in_some_lb_eq_no_jd(self):
            r_ed, r_bol , r_boj = self.run_op(pw_le_some_equal, pw_lib_some_in,pw_jds,  pw_eds, 0.0)
            self.assertEqual(4, self.ex_sum(r_bol) ) # only four, because unlabeled ones are ignored
            self.assertEqual(2, self.ex_sum(r_boj) ) # all jd are greater than zero                
            self.assertTrue(np.array_equal(np.array([[-1,0.1,0.7],[0.1,-1,0.6],[0.7,0.6,-1]], dtype="float32"),r_ed) ) # all euclidean distances should be passed through
        
        def test_some_in_some_lb_eq_jd_greater(self):
            r_ed, r_bol , r_boj = self.run_op(pw_le_some_equal, pw_lib_some_in,pw_jds,  pw_eds, 0.5)
            self.assertEqual(4, self.ex_sum(r_bol) ) # only four, because unlabeled ones are ignored
            self.assertEqual(0, self.ex_sum(r_boj) ) # because all no jd is smaller than 0.0                  
            self.assertTrue(np.array_equal(np.array([[-1,-1.0,0.7],[-1.0,-1.0,0.6],[0.7,0.6,-1]], dtype="float32"),r_ed) ) # all euclidean distances should be passed through


        def tearDown(self):
            self.session.close()
               
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromModule(TestAnchorNegativeExamples()))
    unittest.TextTestRunner().run(suite)