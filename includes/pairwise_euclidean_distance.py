def pairwise_euclidean_distances(
    x1,  # x1 is a 2d tensor of dimension [nr1,b] h_t
    x2,  # x2 is a 2d tensor of dimension [nr2,d] h_s
    result_dtype=tf.float32
):
    x1 = tf.cast(x1, tf.float64) # perhaps cast to 
    x2 = tf.cast(x2, tf.float64)
    
    with tf.variable_scope("PairwiseEuclideanDistance"):
                
        x1_row_norm = tf.reduce_sum(tf.pow(x1,2), axis=1, keep_dims=True) # [n_x1_rows, 1]
        x2_row_norm = tf.reduce_sum(tf.pow(x2,2), axis=1, keep_dims=True) # [n_x2_rows, 1]

        squared_distances=tf.matmul(
            a=x1,
            b=x2,
            transpose_a=False,
            transpose_b=True,
        ) # => [n_x1_rows, n_x2_rows]
        squared_distances = -2 * squared_distances 
        squared_distances = squared_distances + x1_row_norm # => broadcast as row vector 
        pairwise_sqrd_euclidean_distances = tf.abs(squared_distances + tf.transpose(x2_row_norm)) # => broadcast as column vector; 
        # use tf abs, because pairwise sqrd can get small negative zero values
        #pairwise_sqrd_euclidean_distances = tf.abs(pairwise_sqrd_euclidean_distances) # because tensorflow knows -0 for very small numerical values
        pairwise_euclidean_distances = tf.sqrt(pairwise_sqrd_euclidean_distances)        
       
        return tf.cast(pairwise_euclidean_distances, result_dtype), tf.cast(pairwise_sqrd_euclidean_distances, result_dtype)