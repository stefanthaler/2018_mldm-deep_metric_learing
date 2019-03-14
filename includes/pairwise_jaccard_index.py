# https://en.wikipedia.org/wiki/Jaccard_index


# returns a 2d tensor of dimension [nr1,nr2]
# where each element contains the pairwise jaccard index of the respective row vectors of x1 and x2
# so the element result[0,0] is the pairwise jaccard index of x1[0,:] and x2[0,:], that is the first row vector of x1 and x2

def pairwise_jaccard_index(
    x1,  # x1 is a 2d tensor of dimension [nr1,b]
    x2,  # x2 is a 2d tensor of dimension [nr2,d]
):
    with tf.variable_scope("PairwiseJaccardIndex"):
        n_x1_rows = tf.shape(x1)[0] # [nr1, seq_len]
        n_x2_rows = tf.shape(x2)[0] # [nr2, seq_len]
        # first we create a copy for each element row of x1 for each row of x2
        #
        # Example: 
        # x1 has two rows, rx1_1 and rx1_2 
        # x2 has three rows, rx2_1, rx2_2, rx2_3
        #
        # we want two 3d tensors, x1_tiled and x2_tiled that contain:
        # x1_tiled: [rx1_1, rx1_1,rx1_1, rx1_2, rx1_2, rx1_2] 
        # x2_tiled: [rx2_1, rx2_2,rx2_3, rx2_1, rx2_2, rx2_3] 
        # so that we can calculate the pairwise intersection  / union between each of these elements
        x1_expanded = tf.expand_dims(x1,1) # => [nr1,1,b] 
        x1_tiled = tf.tile( 
            input=x1_expanded, 
            multiples=[1, n_x2_rows, 1], 
        ) # => [nr1, nr2, b ]
        # 
        x2_expanded = tf.expand_dims(x2,0) # => [1, nr2, d] 
        x2_tiled = tf.tile( 
            input=x2_expanded, 
            multiples=[n_x1_rows,1, 1]
        )  # => => [nr1, nr2, d ]
        
        # new_shape = tf.shape(tf.transpose(y_row)) # [batch_size, 1]
        # y_row_ary = tf.tile(input=y_row, multiples=new_shape ) # => [batch_size, batchtsize]

        # calculate intersection 
        # we ignore zeros, because they are the padding elements   
        sparse_intersection = tf.sets.set_intersection(x1_tiled,x2_tiled) # 
        dense_intersection = tf.sparse_tensor_to_dense(sparse_intersection) 
        len_intersection = tf.count_nonzero( 
            input_tensor=dense_intersection,
            axis=2,
            keep_dims=False,
            dtype=tf.int32,
        ) # =>  [nr1, nr2]
        
       
        # calculate union
        sparse_union = tf.sets.set_union(x1_tiled, x2_tiled ) # sparse_tensor
        dense_union = tf.sparse_tensor_to_dense(sparse_union) # [nr1, nr2,  _ ]
        len_union = tf.count_nonzero( 
            input_tensor=dense_union,
            axis=2,
            keep_dims=False,
            dtype=tf.int32,
        ) # => [nr1, nr2]

        # get dice coefficent
        pairwise_dice_index = (len_intersection) / (len_union) # => [nr1, nr2]
        pairwise_jaccard_indices =  tf.cast(1 - pairwise_dice_index, tf.float32) # => [nr1, nr2]
        return pairwise_jaccard_indices