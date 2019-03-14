# https://stackoverflow.com/questions/41806689/tensorflow-get-indices-of-array-rows-which-are-zero
# get pairwise jaccard indizes for all samples in the batch


"""
    Get all anchor-positive / anchor-negative example permuations in  batch
"""
def get_positive_and_anchor_examples(
                                    input_x, # input sequences 
                                    input_z, # encoded input sequences in embedding space z
                                    labels_y, # labels for input sequences
                                    jd_pos_threshold=JD_POS_THRESHOLD, jd_neg_threshold=JD_NEG_THRESHOLD):
    # get pairwise jaccard indices for input sequences
    batch_size = tf.shape(input_x)[0]
    
    # get pairwise jaccard distances for input sequences
    pw_ji = pairwise_jaccard_indices(x1=input_x, x2=input_x)  # => [BATCH_SIZE, BATCH_SIZE]
  
    
    # get pairwise euclidean distances for encoded input sequneces
    _ , pw_sq_ed = pairwise_euclidean_distances(x1=input_z, x2=input_z)  # => [BATCH_SIZE, BATCH_SIZE]

    # get pairwise label equality
    pw_lbl_eq = pairwise_label_equality(labels_y)
    pw_lbl_eq = tf.cast(pw_lbl_eq, tf.int32) # => [BATCH_SIZE, BATCH_SIZE] 
        
    # check whether labels are in batch
    labels_in_batch = pairwise_labels_in_batch(labels_y)
   
    # get anchor-positive examples and anchor-negative examples
    anchor_positive_ed, pos_because_of_labels, pos_because_of_jd = anchor_positive_examples(pw_lbl_eq, labels_in_batch, pw_ji, pw_sq_ed, jd_pos_threshold)
    anchor_negative_ed, neg_because_of_labels, neg_because_of_jd = anchor_negative_examples(pw_lbl_eq, labels_in_batch, pw_ji, pw_sq_ed, jd_pos_threshold)

    # get all combinations between a=>p, a=>n for each row 
    # example: assume positive_ed = [[a,b], [c,d]] and negative_ed = [[e,f],[g,h]]
    # then pos_row =
    # [
    #   [a,a],
    #   [b,b],
    #   [c,c],
    #   [d,d]
    # ]
    # and neg_col = 
    # [
    #   [e,f],
    #   [e,f],
    #   [g,h],
    #   [g,h]
    # ]
    # 
    # if you use this arrays for comparison, you will have all possible combinations within one row. 
    # if you calculate the distance between pos_row and neg_col, you will get:
    # [
    #   [a-e, a-f], 
    #   [b-e, b-f], 
    # 
    #  ...
    # ]   
    pos_row = tf.tile(tf.reshape(anchor_positive_ed, [-1, 1]), [1, batch_size])
    neg_col = tf.reshape(tf.tile(anchor_negative_ed, [1 , batch_size]), [-1, batch_size])
    
    # get statistics on how many examples where anchor-positive and how many were anchor negative
    num_neg_la = tf.reduce_sum(tf.cast(neg_because_of_labels, dtype=tf.int32)) 
    num_neg_jd = tf.reduce_sum(tf.cast(neg_because_of_jd, dtype=tf.int32))
    num_pos_la = tf.reduce_sum(tf.cast(pos_because_of_labels, dtype=tf.int32)) 
    num_pos_jd = tf.reduce_sum(tf.cast(pos_because_of_jd, dtype=tf.int32)) 
    stats = (num_neg_la, num_neg_jd, num_pos_la, num_pos_jd)
    
    return pos_row, neg_col, stats