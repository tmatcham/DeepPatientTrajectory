import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from competing_risks.generate_synthetic_competing_risk_data import generate_data

data_path = "D:/PycharmProjects/EHR_survival_prediction/competing_risks/data/"
work_dir = "D:/PycharmProjects/EHR_survival_prediction/competing_risks/"

def competing_exponential_layer(x):
    """
    Lambda function for generating competing exponential parameters
    p1 and p2 from a Dense(2) output.
    Assumes tensorflow 2 backend.

    Usage
    -----
    outputs = Dense(2)(final_layer)
    distribution_outputs = Lambda(competing_exponential_layer)(outputs)

    Parameters
    ----------
    x : tf.Tensor
        output tensor of Dense layer

    Returns
    -------
    out_tensor : tf.Tensor

    """

    # Get the number of dimensions of the input
    num_dims = len(x.get_shape())

    # Separate the parameters
    p1, p2 = tf.unstack(x, num=2, axis=-1)

    # Add one dimension to make the right shape
    p1 = tf.expand_dims(p1, -1)
    p2 = tf.expand_dims(p2, -1)

    # Apply a softplus to make positive
    p1 = tf.keras.activations.softplus(p1)
    p2 = tf.keras.activations.softplus(p2)

    # Join back together again
    out_tensor = tf.concat((p1, p2), axis=num_dims - 1)

    return out_tensor

def competing_exponential_loss(y_true, y_pred):
    """
    Competing exponential loss function.
    Assumes tensorflow backend.

    Parameters
    ----------
    y_true : tf.Tensor
        t and r Ground truth values of predicted variable.
    y_pred : tf.Tensor
        y1 and y2 values of predicted distribution.

    Returns
    -------
    nll : tf.Tensor
        Negative log likelihood.
    """

    #separate the labels
    t, r = tf.unstack(y_true, num=2, axis=-1)

    # Add one dimension to make the right shape
    t = tf.expand_dims(t, -1)
    r = tf.expand_dims(r, -1)

    y1, y2 = tf.unstack(y_pred, num=2, axis=-1)

    # Add one dimension to make the right shape
    y1 = tf.expand_dims(y1, -1)
    y2 = tf.expand_dims(y2, -1)

    nll = (
                (y1*t +y2*t) * tf.cast(tf.equal(r,0.0), tf.float32)
            +   (y1*t +y2*t - tf.math.log(y1 + 0.0000000000000000001)) * tf.cast(tf.equal(r,1.0), tf.float32)
            +   (y1*t +y2*t - tf.math.log(y2 + 0.0000000000000000001)) * tf.cast(tf.equal(r,2.0), tf.float32)
           )
    return nll

def prepare_data(data_path):
    vars_all = np.genfromtxt(data_path+"vars_all.csv", delimiter=',')
    t = np.genfromtxt(data_path+"t_obs.csv", delimiter=',')
    r = np.genfromtxt(data_path+"r_obs.csv", delimiter=',')
    t.shape = (t.shape[0], 1)
    r.shape = (r.shape[0], 1)
    labels = np.concatenate((t,r), axis = 1)

    train_index = int(vars_all.shape[0]*0.8)

    train_examples = vars_all[0:train_index,:]
    test_examples = vars_all[train_index:,:]
    train_labels = labels[0:train_index,:]
    test_labels = labels[train_index:,:]

    train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

    BATCH_SIZE = 64
    SHUFFLE_BUFFER_SIZE = 100

    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    return train_dataset, test_examples, train_index
def swapPositions(list, pos1, pos2):
    list[pos1], list[pos2] = list[pos2], list[pos1]
    return list

def c_concordance(x, y):
    a1 = np.argsort(x, kind = 'quicksort', axis = 0)
    y1 = [float(y[int(i)]) for i in a1]
    N = len(y1)
    stop = 0
    count = 0
    while stop ==0:
        stop = 1
        for i in range(1,N):
            if y1[i-1]>y1[i]:
                swapPositions(y1, i-1, i)
                stop = 0
                count += 1
    conc = 1-count/(N*(N-1)/2)
    return conc
import random
def est_concordance(x, y, s= 50000):
    n = len(x)
    ind = random.sample(range(n*n), s)
    a = [i%n for i in ind]
    b = [i//n for i in ind]
    count = 0
    for i1,i2 in enumerate(a):
        i = a[i1]
        j = b[i1]
        c = x[i] - x[j]
        d = y[i] - y[j]
        if c*d>0:
            count += 1
    return count/s


for i in range(1,6):
    n_prop_relevent = i*(1/5)
    generate_data(data_path, n_patient = 200000 ,n_cat_var = 20, n_cts_var = 20,
                      censored_p = 1, n_prop_relevent=n_prop_relevent, n_risks = 2)
    train_dataset, test_examples, train_index = prepare_data(data_path)

    inputs = tf.keras.Input(shape=(    40,))
    x = tf.keras.layers.Dense(20, activation=tf.nn.leaky_relu)(inputs)
    outputs = tf.keras.layers.Dense(2, activation=tf.nn.softplus)(x)
    #distribution_outputs = tf.keras.layers.Lambda(competing_exponential_layer)(outputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.02),
                  loss= competing_exponential_loss)

    model.fit(train_dataset, epochs=5, verbose=2)


    pred_params = model.predict(test_examples)
    haz_rate = np.genfromtxt(data_path + "haz_rate_array.csv", delimiter=',')
    haz_rate = haz_rate[train_index:, :]

    pred1 = pred_params[:,0]
    pred2 = pred_params[:,1]
    y1 = haz_rate[:, 0]
    y2 = haz_rate[:, 1]



    fig = plt.scatter(pred1, y1, s = 1)
    plt.plot([0,250], [0,250], color='k', linestyle='-', linewidth=2)
    plt.scatter(pred2, y2, s=1)
    plt.savefig(work_dir+"figures/errors_"+str(i)+'.png')
    plt.close()


    print(est_concordance(pred1, y1))
    print(est_concordance(pred2, y2))
    print((est_concordance(pred1, y1)+est_concordance(pred2, y2))/2)


