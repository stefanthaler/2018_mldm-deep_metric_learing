# checks for each combination of an 1d array which labels are in the batch and which ones are not
def pairwise_labels_in_batch(labels):
    batch_size = labels.shape[0]
    assert len(labels.shape.as_list()) == 1, "expect labels to be a 1d tensor of ints of batch_size"
    assert labels.dtype == tf.int32 or labels.dtype==tf.int64, "expect labels to be a 1d tensor of ints of length batch_size"

    y_row = tf.expand_dims(labels,0)
    new_shape = tf.shape(tf.transpose(y_row)) # [batch_size, 1]
    y_row_ary = tf.tile(input=y_row, multiples=new_shape ) # => [batch_size, batchtsize]
    labels_in_batch = tf.logical_and(# IN BATCH
        tf.greater_equal(x=y_row_ary, y=tf.zeros_like(y_row_ary, dtype=y_row_ary.dtype)), 
        tf.greater_equal(x=tf.transpose(y_row_ary), y=tf.zeros_like(y_row_ary, dtype=y_row_ary.dtype)) 
    )
    return labels_in_batch