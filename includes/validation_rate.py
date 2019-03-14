def evaluate_shard(out_csv_name, pw_ji, labels_x, labels_y, d = 0.00, d_step = 0.005, d_max=1.0):
    
    h.save_to_csv(data_rows=[[
        "Distance Threshhold",
        "True Positives", 
        "False Positives", 
        "True Negative", 
        "False Negative", 
        "Num True Same", 
        "Num True Diff", 
    ]], outfile_name=out_csv_name, mode="w")
    
    
    # calculate true accepts / false accepts based on labels
    n_labels = len(labels_x)
    tl_row = np.repeat( np.array(labels_x).reshape((n_labels,1)), n_labels, axis=1 )
    tl_col = np.repeat( np.array(labels_y).reshape((1,n_labels)), n_labels, axis=0 ) 
    p_same = np.equal(tl_row, tl_col).astype("int8")
    p_diff = np.not_equal(tl_row, tl_col).astype("int8")
    num_true_same = p_same.sum()
    num_true_diff = p_diff.sum()
    
    while True:
        calc_same = np.zeros((n_labels, n_labels))
        calc_same[np.where(pw_ji<=d)]=1
        
        tp = np.sum(np.logical_and(calc_same, p_same))
        fp = np.sum(np.logical_and(calc_same, np.logical_not(p_same)))
        tn = np.sum(np.logical_and(np.logical_not(calc_same), np.logical_not(p_same)))
        fn = np.sum(np.logical_and(np.logical_not(calc_same), p_same))
        
        h.save_to_csv(data_rows=[[d, tp, fp, tn, fn,num_true_same,num_true_diff]], outfile_name=out_csv_name, mode="a")
        
        d+=d_step
        if d>d_max:
            break

def evaluate_all_shards(inputs, labels, shard_size,shard_indizes,  results_fn, d_start=0.0, d_step=0.005, d_max=1.0 ):
    for shard_index in shard_indizes:
        shard_x, shard_y = shard_index
        print("Current shard", shard_index)
        start_index_x = shard_x*shard_size
        start_index_y = shard_y*shard_size
        end_index_x = min((shard_x+1)*shard_size, num_test_examples)
        end_index_y = min((shard_y+1)*shard_size, num_test_examples)

        # calcualte pairwise distances
        shard_inputs_x = inputs[start_index_x:end_index_x,:]
        shard_labels_x = labels[start_index_x:end_index_x]

        shard_inputs_y = inputs[start_index_y:end_index_y,:]
        shard_labels_y = labels[start_index_y:end_index_y]

        pw_ji = pairwise_distances(shard_inputs_x,shard_inputs_y, metric=jaccard_distance, n_jobs=8) 

        # evaluate pairwise distances 
        out_csv_name = results_fn+"_%0.2d-%0.2d"%(shard_x, shard_y)
        evaluate_shard(out_csv_name, pw_ji, shard_labels_x, shard_labels_y, d=d_start,  d_step = d_step, d_max=d_max)            
            
def run_evaluation(inputs, labels, shard_size, results_fn, d_start=0.0, d_step=0.005, d_max=1.0):
    results_fn = results_fn%shard_size
    
    num_test_examples = inputs.shape[0]
    num_x = inputs.shape[0]//shard_size
    if not num_test_examples%shard_size==0 :# need to be a square matrix
        print("Allowed shard sizes")
        for i in range(100, num_test_examples):
            if num_test_examples%i==0:
                print(i)
        0/0
    shard_indizes = list(itertools.product(range(num_x),repeat=2))
    num_shards = len(shard_indizes)
    num_distances = len(list(np.arange(d_start,d_max,d_step)))
    num_metrics = 7 
    
    evaluate_all_shards(inputs, labels, shard_size, shard_indizes, results_fn, d_start, d_step, d_max )
    
    all_data = np.ndarray(shape=(num_shards, num_distances, num_metrics), dtype="float32")

    for i, shard_index in enumerate(shard_indizes):
        # load shard
        shard_x, shard_y = shard_index
        out_csv_name = results_fn+"_%0.2d-%0.2d"%(shard_x, shard_y)
        shard_data = h.load_from_csv(out_csv_name)
        shard_data = shard_data[1:] # cut header row 
        all_data[i] = np.array(shard_data)


    final_data  = np.ndarray(shape=(num_distances, 10), dtype="float32")

    final_data[:,0] = all_data[0,:,0] # all distances (are same over all shards)

    final_data[:,1] = all_data.sum(axis=0)[:,1] # True Positives
    final_data[:,2] = all_data.sum(axis=0)[:,2] # False Positives
    final_data[:,3] = all_data.sum(axis=0)[:,3] # True Negatives
    final_data[:,4] = all_data.sum(axis=0)[:,4] # False Negatives
    final_data[:,5] = all_data.sum(axis=0)[:,5] # Num true same (are same over all shards)
    final_data[:,6] = all_data.sum(axis=0)[:,6] # Num true diff  (are same over all shards)

    final_data[:,7] = final_data[:,1]/final_data[:,5] # validation rate 
    final_data[:,8] = final_data[:,2]/final_data[:,6] # false acceptance rate  

    final_data[:,9] = (final_data[:,1] + final_data[:,3]) / (final_data[:,1:1+4].sum(axis=1)) 

    
    h.save_to_csv(data_rows=[[
            "Distance Threshhold",
            "True Positives", 
            "False Positives", 
            "True Negative", 
            "False Negative", 
            "Num true same", 
            "Num true diff", 
            "Validation Rate",
            "False Acceptance Rate",
            "Accuracy"
        ]], outfile_name=results_fn, mode="w", convert_float=False)
    h.save_to_csv(data_rows=final_data, outfile_name=results_fn, mode="a", convert_float=True)

    logger.info("Evaluation done, saved to '%s'"%results_fn)
    return final_data