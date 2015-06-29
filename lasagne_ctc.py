import numpy as np
import pickle
import theano
import theano.tensor as T
import lasagne
import gzip

#from ctc_cost import CTC

n_epochs = 200
num_hidden = 100
MODEL_SEQ_LEN = 700
BATCH_SIZE = 16

NUM_CLASSES = len("XMIO")
NUM_INPUTS = len("ARNDCQEGHILKMFPSTWYVX")


def print_pred(y_hat):
    blank_symbol = NUM_CLASSES
    res = []
    for i, s in enumerate(y_hat):
        if (s != blank_symbol) and (i == 0 or s != y_hat[i - 1]):
            res += [s]
    return np.asarray(res)


def load_training_data(val_split, test_split, model_seq_len, batch_size):
    # data is formatted aa, tar, name, input_mask, target_mask

    train_split = [i for i in range(10) if i not in [val_split, test_split]]

    with open("ctc_data.pkl", "rb") as pkl_file:
        data = pickle.load(pkl_file)
    val_data = data[val_split]
    test_data = data[test_split]

    train_data = []
    for i in train_split:
        train_data += data[i]
    return train_data, val_data, test_data


train_data, val_data, test_data = load_training_data(
    0, 1, MODEL_SEQ_LEN, BATCH_SIZE)


def create_train_batch():
    """
    Converts the data to batches.

    """
    shuffle = np.random.permutation(BATCH_SIZE)
    batch = [train_data[s] for s in shuffle]

    # data is formatted aa, tar, tar_ctc, name, input_mask, target_mask
    input = np.concatenate([map(lambda x: x[0], batch)]).astype('float32')
    labels = np.concatenate([map(lambda x: x[1], batch)]).astype('float32')
    labels_ctc = np.concatenate([map(lambda x: x[2], batch)]).astype('float32')
    input_mask = np.concatenate([map(lambda x: x[4], batch)]).astype('float32')
    labels_ctc_mask = np.concatenate(
        [map(lambda x: x[5], batch)]).astype('float32')
    return input, labels, labels_ctc, input_mask, labels_ctc_mask

# S = 100   (num batches)
# T = 30 seqlen
# B = 10 batchsize
# D = 4, inputdim
# L = target length, varies in the example....

# inputs      :  S x T x B x inputdim    (num_batches, seq_len, batch_size, input_dim)
# inputs_mask :  S x T x B               (num_batches, seq_len, batch_size)
# labels      :  L x B   (L varies probably because the sequences arex = tensor.tensor3('x', dtype=floatX)
# T x B


x_sym = T.matrix('x', dtype='float32')           # B x T x F   # only matrix because i use embedding...
x_mask_sym = T.matrix('x_mask', dtype='float32')  # B x T
y_sym = T.matrix('y', dtype='float32')            # B x L
y_mask_sym = T.matrix('y_mask', dtype='float32')  # B x L
y_hat_mask_sym = x_mask_sym
y_hat_sym = T.tensor3('y_hat', dtype='float32')    # B x L x C+1

l_inp = lasagne.layers.InputLayer((BATCH_SIZE, MODEL_SEQ_LEN))
l_emb = lasagne.layers.EmbeddingLayer(
    l_inp, input_size=NUM_INPUTS, output_size=50)
l_rec = lasagne.layers.LSTMLayer(l_emb, num_units=num_hidden)
l_shp = lasagne.layers.reshape(l_rec, (BATCH_SIZE*MODEL_SEQ_LEN, num_hidden))

l_softmax = lasagne.layers.DenseLayer(l_shp, num_units=NUM_CLASSES,
                                      nonlinearity=T.nnet.softmax)

l_out = lasagne.layers.reshape(l_softmax,
                               (BATCH_SIZE, MODEL_SEQ_LEN, NUM_CLASSES))

# The input format to CTC is seq_len, batch_size, num_inputs
output = lasagne.layers.get_output(l_out, T.cast(x_sym, 'int32'))

output_eval = lasagne.layers.get_output(l_out, T.cast(x_sym, 'int32'),
                                        deterministic=True)


eval = theano.function([x_sym], output_eval)
#input, labels, labels_ctc, input_mask, labels_ctc_mask = create_train_batch()
#print eval(input).shape
#print "Eval...Done"

all_params = lasagne.layers.get_all_params(l_out, trainable=True)
output_flat = T.reshape(output, (BATCH_SIZE*MODEL_SEQ_LEN, NUM_CLASSES))
cost = T.nnet.categorical_crossentropy(output_flat,
                                       T.cast(y_sym.flatten(), 'int32'))
cost = T.mean(cost)
#cost = CTC().apply(y_sym, output, y_mask_sym, y_hat_mask_sym,
#                   'log_scale')
all_grads = T.grad(cost, all_params)
updates = lasagne.updates.rmsprop(all_grads, all_params, learning_rate=0.001)
train = theano.function([x_sym, y_sym],
                        [cost, output], updates=updates)




#def create_val_batches():
#    n_val_batches = len(val_data) // BATCH_SIZE
#    n_val_samples = n_val_batches*BATCH_SIZE
#    data = val_data[:n_val_samples]
#
#     batches = []
#     for i in range(n_val_batches):
#
#         batch_idx = [train_data[s] for s in range(i*BATCH_SIZE, (i+1)*BATCH_SIZE)]
#
#         # data is formatted aa, tar, name, input_mask, target_mask
#         input = np.concatenate(
#             [map(lambda x: x[0], batch_idx)]).astype('float32')
#         labels = np.concatenate(
#             [map(lambda x: x[1], batch_idx)]).astype('float32')
#         input_mask = np.concatenate(
#             [map(lambda x: x[3], batch_idx)]).astype('float32')
#         labels_mask = np.concatenate(
#             [map(lambda x: x[4], batch_idx)]).astype('float32')
#
#         batches.append([input, labels, input_mask, labels_mask])
#
#
#     return batches
#
#
#test = create_val_batches()

print "Training"
cost_lst = []
epoch = 0
for i in range(1000000):
    if (i %25) == 0:
        if len(cost_lst) > 0:
            epoch += 1
            print "--------EPOCH--------", epoch
            print np.mean(cost_lst)
            cost_lst = []

    (input, labels,
     labels_ctc,
     input_mask,
     labels_ctc_mask) = create_train_batch()
    cst, prd = train(input, labels)
    cost_lst.append(cst)


    # if (i%200) == 0:
    #     print "X"*25
    #     val_batches = create_val_batches()
    #     for vb in val_batches:
    #         input, labels, input_mask, labels_mask = vb
    #         prd = eval(input)
    #
    #         for j in range(BATCH_SIZE):
    #             print "".join(map(str, list(print_pred(np.argmax(prd[j], axis=-1)))))
    #             print "".join(map(lambda x: str(int(x)), list(labels[j])))
    #             print "-"*25

    # data[i%batch_size].astype('float32'),
    #          labels[i%batch_size].astype('float32'),
    #          labels_mask[i%batch_size].astype('float32'),
    #          inputs_mask[i%batch_size].astype('float32')))

