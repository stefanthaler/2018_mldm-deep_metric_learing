# returns all euclidean distances for which either the labels are the same or the jaccard distance is below the positive threshold
def anchor_positive_examples(pw_label_equality, labels_in_batch, pw_jaccard_distances, pw_euclidean_distances, jd_pos_threshold):
    batch_size = tf.shape(pw_label_equality)[0]
    labels_not_in_batch = tf.logical_not(labels_in_batch) # labels in batch is a bad name. It should be - we have labeled exampled for this example
    
    # positive conditions
    labels_match = tf.not_equal(pw_label_equality, tf.eye(batch_size, dtype=tf.int32)) # exclude equality between same elements
    pw_ji_for_pos = tf.add(pw_jaccard_distances, tf.eye(batch_size)*1.5) # jaccard distance is between 0 and 1 - exclude equality between same elements
    sequences_have_pos_jd = tf.less(x=pw_ji_for_pos, y=jd_pos_threshold, name="jd_pos_cond") # sequences are
    
    # it's either an anchor-positive example because the jaccard distance is smaller than the threshold or because the labels are the same. 
    pos_because_of_labels = tf.logical_and(labels_in_batch, labels_match)
    pos_because_of_jd  = tf.logical_and(labels_not_in_batch, sequences_have_pos_jd)
    pos_cond = tf.logical_or(pos_because_of_labels, pos_because_of_jd)
    
    # exclude example itself from positive / negative  - euclidean distance to between two identical vectors should always be 0
    positive_ed = tf.where(condition=pos_cond , x=pw_euclidean_distances, y=tf.ones_like(pw_euclidean_distances)*-1) # -1 means non positive
    return positive_ed, pos_because_of_labels,  pos_because_of_jd