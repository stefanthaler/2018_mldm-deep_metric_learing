def anchor_negative_examples(pw_label_equality, labels_in_batch, pw_jaccard_distances, pw_euclidean_distances, jd_neg_threshold):
    batch_size = tf.shape(pw_label_equality)[0]
    labels_not_in_batch = tf.logical_not(labels_in_batch)
    
    # make sure to exclude jaccard distances of the diagonal, because elements to itself are never negative
    pw_ji_for_neg = tf.add(pw_jaccard_distances,  tf.eye(batch_size)*-1.0)
    
    # negative condition
    labels_dont_match = tf.equal(pw_label_equality, tf.zeros_like(pw_label_equality, dtype=tf.int32),  name="la_neg_cond")
    sequences_have_neg_jd = tf.greater_equal(x=pw_ji_for_neg, y=jd_neg_threshold, name="jd_neg_cond") # elements at the diagonal should aways have zero, so 
    
    neg_because_of_labels = tf.logical_and(labels_in_batch, labels_dont_match) # all labels that are not equal to 1
    neg_because_of_jd = tf.logical_and(labels_not_in_batch, sequences_have_neg_jd)  
    neg_cond = tf.logical_or(neg_because_of_labels, neg_because_of_jd) # it's either negative because the jaccard distance is over the threshold or the labels are not matching                                           
    
    negative_ed = tf.where(condition=neg_cond , x=pw_euclidean_distances, y=tf.ones_like(pw_euclidean_distances)*-1)
    return negative_ed, neg_because_of_labels, neg_because_of_jd