# Encoder
def LSTMEncode(input_x, input_sequences_length, scope, name=""):
    input_sequences = tf.nn.embedding_lookup(TOKEN_EMBEDDINGS, input_x) # [BATCH_SIZE, max_time, embedding_size]
    
    encoder_cell = tf.contrib.rnn.LSTMCell(num_units=STATE_SIZE, state_is_tuple=True)
    encoder_cell = tf.contrib.rnn.DropoutWrapper(cell=encoder_cell,
                                         output_keep_prob=TF_KEEP_PROBABILTIY,
                                         input_keep_prob=TF_KEEP_PROBABILTIY,
                                         state_keep_prob=TF_KEEP_PROBABILTIY,
                                         dtype=DTYPE)
    encoder_cell = tf.contrib.rnn.MultiRNNCell(cells=[encoder_cell] * NUM_LSTM_LAYERS, state_is_tuple=True)

    encoder_outputs, last_encoder_state = tf.nn.dynamic_rnn(
        cell=encoder_cell,
        dtype=DTYPE,
        sequence_length=input_sequences_length,
        inputs=input_sequences,
        scope=scope
        )
    last_c, last_h = last_encoder_state[0] # h is hidden state, c = memory state https://arxiv.org/pdf/1409.2329.pdf
    z = tf.nn.l2_normalize(x=last_c, dim=1, epsilon=1e-12, name="OutputNormalization")   
    return z     

with tf.variable_scope("Encode_Inputs") as encode_scope:
    z_jd = LSTMEncode(x_jd, x_jd_seq_lengths, encode_scope)