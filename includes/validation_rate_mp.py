import multiprocessing
from multiprocessing import Pool
np.set_printoptions(precision=3, suppress=True)

s2 = time.time()
# multi processing stuff
d_step = 0.01
num_processes = 8
results = {} # shared dictionary to write results to 
pool = Pool(processes=num_processes) # start 8 worker processes

pw_euclid = euclidean_distances(X=le, Y=le).astype("float16")
print("Calculated pairwise euclidean distances")
# calculate true accepts / false accepts based on labels
n_labels = len(true_labels)
tl_row = np.repeat( np.array(true_labels).reshape((n_labels,1)), n_labels, axis=1 )
tl_col = np.repeat( np.array(true_labels).reshape((1,n_labels)), n_labels, axis=0 ) 
p_same = np.equal(tl_row, tl_col).astype("int8")
p_diff = np.not_equal(tl_row, tl_col).astype("int8")
num_true_same = p_same.sum()
num_true_diff = p_diff.sum()


# calculate true accepts / false accepts based on euclidean distances for one example
def calc_vf_rate(distance_threshold):
    calc_same = np.zeros((n_labels, n_labels))
    calc_same[np.where(pw_euclid<=distance_threshold)]=1

    ta_d = np.logical_and(calc_same, p_same).astype("int8")        
    fa_d = np.logical_and(calc_same, p_diff).astype("int8")

    val_d = ta_d.sum() / num_true_same
    far_d = fa_d.sum() / num_true_diff

    return (distance_threshold, val_d, far_d)

# loop to calculate all valid / false acceptance rates until abort condition is fullfilled 
iteration=0
thresholds = np.arange(0.0, 20.0, d_step)

while True and iteration<1000:
    # calculate numprocess steps 
    start_idx = iteration * num_processes
    end_idx = (iteration+1) * num_processes
    step_results = pool.map(calc_vf_rate, thresholds[start_idx:end_idx])
    for sr in step_results:
        results[sr[0]]=sr[1:]
    
    # abort condition
    max_far = np.max([vf[1] for vf in results.values()])
    print("Max FAR", max_far)
    if (int(np.round(max_far,3))==1):
        break

    # get new step range
    iteration+=1

# clean up
pool.close()
pool.join()

# sort results
valid_accepts = []
false_accepts = []
for t in sorted(results.keys()):
    val_d, far_d = results[t]
    valid_accepts.append(val_d)
    false_accepts.append(far_d)
    if (int(np.round(far_d,3))==1):
        break 
        
e2 = time.time()