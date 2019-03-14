import tensorflow as tf
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# check whether some gradient can flow to your trainable parameters
# rudimentary check for gradient flow
def ensure_gradient_flow(operations):
    tf_params = tf.trainable_variables()

    no_flows = 0
    for op in operations:
        gradients = tf.gradients(op, tf_params)
        at_least_one_gradient_flows = False
        for c in zip(gradients, tf_params):
            if not type(c[0]).__name__=="NoneType":
                at_least_one_gradient_flows = True
                break
        if not at_least_one_gradient_flows:
            logger.warn("No gradient flow for operation '%s'"%op.name)
            no_flows += 1
    if no_flows==0:
        logger.info("Operations [%s] have at least 1 gradient for at least 1 parameter"%([o.name for o in operations]))


def print_memory_estimates(max_seq_length=1):
    from tensorflow.python.client import device_lib
    import psutil # pip install psutil
    import numpy as np


    total_num_variables = 0
    total_memory = 0

    local_variables = tf.local_variables()
    global_variables = tf.global_variables()
    all_variables = local_variables+global_variables

    print("Variables:")
    for v in all_variables:
        params = np.prod(v.get_shape().as_list())
        print("\t params: %s, memory: %0.3f in (mbytes) : Name %s"%(str(params).rjust(10), v.dtype.size * params / 1024 /1024, v.name))
        total_num_variables += params
        total_memory += v.dtype.size * params

    local_device_protos = device_lib.list_local_devices()
    print("\nDevices:")
    for dev in local_device_protos:
        print("\t%s : %0.2f (in mbytes)"%(dev.device_type, dev.memory_limit/1024/1024))
    gpu_memory_avail = np.sum([x.memory_limit for x in local_device_protos if x.device_type == 'GPU'])
    cpu_memory_avail = psutil.virtual_memory().total

    print("\nSummary:")
    print("\tTotal model parameters:%i"%total_num_variables)
    if max_seq_length>1:
        print("\tMax sequence length: %i"%max_seq_length)
    print("\tTotal model memory consumption: %i (bytes), ~%0.3f (mbytes)"%(total_memory, total_memory/(1024*1024.0)))
    print("\tGPUs available: %i"%len([x.memory_limit for x in local_device_protos if x.device_type == 'GPU']))
    print("\tTotal GPU memory available: %i (bytes), ~%0.3f (mbytes)"%(gpu_memory_avail, gpu_memory_avail/(1024*1024.0)))
    print("\tTotal CPU memory available: %i (bytes), ~%0.3f (mbytes)"%(cpu_memory_avail, cpu_memory_avail/(1024*1024.0)))
    print("\tMax batch_size CPU: ~ %i examples"%(int(cpu_memory_avail/total_memory)))
    print("\tMax batch_size GPU: ~ %i examples"%(int(gpu_memory_avail/total_memory)))


def link_embedding_to_metadata(embedding_var, metadata_file, graph_dir):
    from tensorflow.contrib.tensorboard.plugins import projector # for visualizing embeddings
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name.replace(":0","")
    embedding.metadata_path = metadata_file
    summary_writer = tf.summary.FileWriter(graph_dir)
    projector.visualize_embeddings(summary_writer, config)
    
def initialize_unitialized_variables(session):
    uninitialized_var_names = set([v.decode('ascii') for v in  session.run(tf.report_uninitialized_variables())])
    uninitialized_vars = [v for v in tf.global_variables() if v.name.split(':')[0] in uninitialized_var_names]
    session.run( tf.variables_initializer( uninitialized_vars ))

# https://en.wikipedia.org/wiki/Jaccard_index


# returns a 2d tensor of dimension [nr1,nr2]
# where each element contains the pairwise jaccard index of the respective row vectors of x1 and x2
# so the element result[0,0] is the pairwise jaccard index of x1[0,:] and x2[0,:], that is the first row vector of x1 and x2

def pairwise_jaccard_indices(
    x1,  # x1 is a 2d tensor of dimension [nr1,b]
    x2,  # x2 is a 2d tensor of dimension [nr2,d]
    n_x1_rows=None, # if nr1 in x1 is unspecified, (e.g., None because the batch size is not specified) you have to specify the number of rows here
    n_x2_rows=None
):
    with tf.variable_scope("PairwiseJaccardIndex"):
        if not n_x1_rows:
            n_x1_rows = tf.cast(x1.shape[0], tf.int64) # nr1
        if not n_x2_rows:
            n_x2_rows = tf.cast(x2.shape[0], tf.int64) # nr2
        
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
        x1_tiled = tf.tile( input=x1_expanded, multiples=[1 , n_x2_rows , 1] ) # => [nr1, nr2, b ]
        x2_expanded = tf.expand_dims(x2,0) # => [1, nr2, d] 
        x2_tiled = tf.tile( input=x2_expanded, multiples=[n_x1_rows,1, 1] )  # => => [nr1, nr2, d ]

        # calculate intersection 
        # we ignore zeros, because they are the padding elements   
        sparse_intersection = tf.sets.set_intersection(x1_tiled,x2_tiled) # 
        dense_intersection = tf.sparse_tensor_to_dense(sparse_intersection) 
        len_intersection = tf.count_nonzero( input_tensor=dense_intersection, axis=2, keep_dims=False, dtype=tf.int64 ) # =>  [nr1, nr2]       
       
        # calculate union
        sparse_union = tf.sets.set_union(x1_tiled, x2_tiled ) # sparse_tensor
        dense_union = tf.sparse_tensor_to_dense(sparse_union) # [nr1, nr2,  _ ]
        len_union = tf.count_nonzero( input_tensor=dense_union, axis=2, keep_dims=False, dtype=tf.int64 ) # => [nr1, nr2]

        # get dice coefficent
        pairwise_dice_index = (len_intersection) / (len_union) # => [nr1, nr2]
        pairwise_jaccard_indices =  tf.cast(1 - pairwise_dice_index, tf.float32) # => [nr1, nr2]
        return pairwise_jaccard_indices

    
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
    
    