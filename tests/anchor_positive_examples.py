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


def run_tests_on(anchor_positive_examples):
    
    class TestAnchorPositiveExamples(TestCase):
        def setUp(self):
            
            self.pw_label_equality=tf.placeholder(shape=[None, None], dtype=tf.int32) # [batch_size, batch_size] 
            self.labels_in_batch=tf.placeholder(shape=[None, None], dtype=tf.bool) # [batch_size, batch_size] 
            self.pw_jaccard_distances= tf.placeholder(shape=[None, None], dtype=tf.float32) # [batch_size, batch_size] 
            self.pw_euclidean_distances= tf.placeholder(shape=[None, None], dtype=tf.float32) # [batch_size, batch_size] 
            self.jd_pos_threshold=tf.placeholder(shape=(), dtype=tf.float32)
            
            self.op = anchor_positive_examples(self.pw_label_equality, self.labels_in_batch, self.pw_jaccard_distances, self.pw_euclidean_distances, self.jd_pos_threshold )

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

        def test_all_in_all_equal_all_jd(self):
            r_ed, r_bol , r_boj = self.run_op(pw_le_all_equal, pw_lib_all_in, pw_jds, pw_eds, 0.9)
            self.assertEqual(6, self.ex_sum(r_bol) ) # all examples labeled, exclude pos to itself
            self.assertEqual(0, self.ex_sum(r_boj) ) # because all examples are labelled, ignore                  
            self.assertTrue(np.array_equal(np.array([[-1.0,0.1,0.7],[0.1,-1.0,0.6],[0.7,0.6,-1.0]], dtype="float32"),r_ed) ) # all euclidean distances should be passed through
        
        def test_all_in_all_equal_no_jd(self):
            r_ed, r_bol , r_boj = self.run_op(pw_le_all_equal, pw_lib_all_in, pw_jds, pw_eds, 0.0)
            self.assertEqual(6, self.ex_sum(r_bol) ) # all examples labeled, exclude pos to itself
            self.assertEqual(0, self.ex_sum(r_boj) ) # because all examples are labelled, ignore                   
            self.assertTrue(np.array_equal(np.array([[-1.0,0.1,0.7],[0.1,-1.0,0.6],[0.7,0.6,-1.0]], dtype="float32"),r_ed) ) # all euclidean distances should be passed through
            
        def test_all_in_non_equal_all_jd(self):
            r_ed, r_bol , r_boj = self.run_op(pw_le_non_equal, pw_lib_all_in, pw_jds, pw_eds, 0.9)
            self.assertEqual(0, self.ex_sum(r_bol) ) # all examples labeled, exclude pos to itself
            self.assertEqual(0, self.ex_sum(r_boj) ) # because all examples are labelled, ignore                  
            self.assertTrue(np.array_equal(np.array([[-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0]], dtype="float32"),r_ed) ) # all euclidean distances should be passed through
        
        def test_all_in_non_equal_no_jd(self):
            r_ed, r_bol , r_boj = self.run_op(pw_le_non_equal, pw_lib_all_in, pw_jds, pw_eds, 0.0)
            self.assertEqual(0, self.ex_sum(r_bol) ) # all examples labeled, exclude pos to itself
            self.assertEqual(0, self.ex_sum(r_boj) ) # because all examples are labelled, ignore                   
            self.assertTrue(np.array_equal(np.array([[-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0]], dtype="float32"),r_ed) ) # all euclidean distances should be passed through
            
        
        def test_non_in_no_all_jd_all_la_eq(self):
            r_ed, r_bol , r_boj = self.run_op(pw_le_all_equal, pw_lib_non_in, pw_jds, pw_eds, 0.9)
            self.assertEqual(0, self.ex_sum(r_bol) ) # ignore, because no examples are labeled 
            self.assertEqual(6, self.ex_sum(r_boj) ) # because all pw jd are smaller than 0.9               
            self.assertTrue(np.array_equal(np.array([[-1.0,0.1,0.7],[0.1,-1.0,0.6],[0.7,0.6,-1.0]], dtype="float32"),r_ed) ) # all euclidean distances should be passed through
        
        def test_non_in_no_two_jd_all_la_eq(self):
            r_ed, r_bol , r_boj = self.run_op(pw_le_all_equal, pw_lib_non_in, pw_jds, pw_eds, 0.8)
            self.assertEqual(0, self.ex_sum(r_bol) ) # ignore, because no examples are labeled 
            self.assertEqual(4, self.ex_sum(r_boj) ) # only two pw jd are smaller than 0.8 (Note: 0.8 < 0.8 = False)                 
            self.assertTrue(np.array_equal(np.array([[-1.0,0.1,-1.0],[0.1,-1.0,0.6],[-1.0,0.6,-1.0]], dtype="float32"),r_ed) ) # all euclidean distances should be passed through
        
        def test_non_in_no_all_jd_no_la_eq(self):
            r_ed, r_bol , r_boj = self.run_op(pw_le_non_equal, pw_lib_non_in, pw_jds, pw_eds, 0.9)
            self.assertEqual(0, self.ex_sum(r_bol) ) # ignore, because no examples are labeled 
            self.assertEqual(6, self.ex_sum(r_boj) ) # because all pw jd are smaller than 0.9               
            self.assertTrue(np.array_equal(np.array([[-1.0,0.1,0.7],[0.1,-1.0,0.6],[0.7,0.6,-1.0]], dtype="float32"),r_ed) ) # all euclidean distances should be passed through
        
        def test_non_in_no_two_jd_no_eq(self):
            r_ed, r_bol , r_boj = self.run_op(pw_le_non_equal, pw_lib_non_in, pw_jds, pw_eds, 0.8)
            self.assertEqual(0, self.ex_sum(r_bol) ) # ignore, because no examples are labeled 
            self.assertEqual(4, self.ex_sum(r_boj) ) # because all examples are labelled                  
            self.assertTrue(np.array_equal(np.array([[-1.0,0.1,-1.0],[0.1,-1.0,0.6],[-1.0,0.6,-1.0]], dtype="float32"),r_ed) ) # all euclidean distances should be passed through
        
        # some examples labeled, others not 
        def test_some_in_all_lb_eq_no_jd(self):
            r_ed, r_bol , r_boj = self.run_op(pw_le_all_equal, pw_lib_some_in,pw_jds,  pw_eds, 0.0)
            self.assertEqual(4, self.ex_sum(r_bol) ) # only four, because unlabeled ones are ignored
            self.assertEqual(0, self.ex_sum(r_boj) ) # because all no jd is smaller than 0.0                  
            self.assertTrue(np.array_equal(np.array([[-1.0,-1.0,0.7],[-1.0,-1.0,0.6],[0.7,0.6,-1.0]], dtype="float32"),r_ed) ) # all euclidean distances should be passed through
        
         # some examples labeled, others not 
        def test_some_in_all_lb_eq_jd_greater(self):
            r_ed, r_bol , r_boj = self.run_op(pw_le_all_equal, pw_lib_some_in,pw_jds,  pw_eds, 0.3)
            self.assertEqual(4, self.ex_sum(r_bol) ) # only four, because unlabeled ones are ignored
            self.assertEqual(2, self.ex_sum(r_boj) ) # because all no jd is smaller than 0.0                  
            self.assertTrue(np.array_equal(np.array([[-1.0,0.1,0.7],[0.1,-1.0,0.6],[0.7,0.6,-1.0]], dtype="float32"),r_ed) ) # all euclidean distances should be passed through


        def tearDown(self):
            self.session.close()
               
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromModule(TestAnchorPositiveExamples()))
    unittest.TextTestRunner().run(suite)