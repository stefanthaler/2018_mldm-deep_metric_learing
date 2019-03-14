import tensorflow as tf

def pairwise_label_equality(labels):
    # check if labels are of correct size and type
    batch_size = labels.shape[0]
    assert len(labels.shape.as_list()) == 1, "expect labels to be a 1d tensor of ints of batch_size"
    assert labels.dtype == tf.int32 or labels.dtype==tf.int64, "expect labels to be a 1d tensor of ints of length batch_size"

    y_row = tf.expand_dims(labels,0) # [1,batch_size]
    new_shape = tf.shape(tf.transpose(y_row)) # [batch_size, 1]
    y_row_ary = tf.tile(input=y_row, multiples=new_shape ) # => [batch_size, batchtsize]
    pw_label_equality = tf.equal(y_row_ary, tf.transpose(y_row_ary))
    return pw_label_equality
