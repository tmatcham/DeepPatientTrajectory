import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import csv

work_dir = 'D:/PycharmProjects/EHR_survival_prediction/single_risk_censored/'
# Define Sequential model with 3 layers
# Call model on a test input

random.seed(1234)

#Import the data:

def exponential_loss(y_true, y_pred):
    """
    Negative binomial loss function.
    Assumes tensorflow backend.

    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth values of predicted variable.
    y_pred : tf.Tensor
        n and p values of predicted distribution.

    Returns
    -------
    nll : tf.Tensor
        Negative log likelihood.
    """

    t, C = tf.unstack(y_true, num=2, axis=-1)
    t = tf.expand_dims(t, -1)
    C = tf.expand_dims(C, -1)

    # Calculate the negative log likelihood
    nll = (
        (y_pred * t -tf.math.log(y_pred+0.00000000000000001)) * tf.cast(tf.math.equal(C, 0.0), tf.float32) +
        (y_pred * t) * tf.cast(tf.math.equal(C, 1.0), tf.float32)
    )
    return nll
def create_model1temp(n_layers = 1, width = 20, drop_p = 0.1):
    inputs = tf.keras.Input(shape=(40,))
    x = tf.keras.layers.Dense(width, activation=tf.nn.leaky_relu)(inputs)
    for i in range(n_layers-1):
        x = tf.keras.layers.Dense(width, activation= tf.nn.leaky_relu)(x)
    x = tf.keras.layers.Dropout(drop_p)(x)
    outputs = tf.keras.layers.Dense(1, activation=tf.nn.softplus)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return(model)

def create_model1(n_layers = 1, width = 20, drop_p = 0.1, input_shape = 40):
    inputs = tf.keras.Input(shape=(input_shape,))
    x = tf.keras.layers.Dense(width, activation=tf.nn.leaky_relu)(inputs)
    for i in range(n_layers-1):
        x = tf.keras.layers.Dense(width, activation= tf.nn.leaky_relu)(x)
    x = tf.keras.layers.Dropout(drop_p)(x)
    outputs = tf.keras.layers.Dense(1, activation=tf.nn.softplus)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return(model)
def est_concordance(x, y, s= 10000):
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



def run_nn(n_patient = 200000, censored_p = 0.8, n_prop_relevent= 0.2):
    X = np.genfromtxt(work_dir+"data/vars_all"+str(n_patient)+'_'+str(censored_p)+'_'+str(n_prop_relevent)+".csv", delimiter=',')
    t = np.genfromtxt(work_dir+"data/times"+str(n_patient)+'_'+str(censored_p)+'_'+str(n_prop_relevent)+".csv", delimiter=',')
    haz_rate = np.genfromtxt(work_dir+"data/haz_rate"+str(n_patient)+'_'+str(censored_p)+'_'+str(n_prop_relevent)+".csv", delimiter=',')
    C = np.genfromtxt(work_dir+"data/obs_ind"+str(n_patient)+'_'+str(censored_p)+'_'+str(n_prop_relevent)+".csv", delimiter=',')

    train_index = int(X.shape[0]*0.8)

    train_X = X[0:train_index,:]
    test_X = X[train_index:,:]

    t.shape = (t.shape[0], 1)
    C.shape = (C.shape[0], 1)
    Y = np.concatenate((t,C), axis = 1)

    train_labels = Y[0:train_index, :]
    test_labels = Y[train_index:, :]

    train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_X, test_labels))

    BATCH_SIZE = 64
    SHUFFLE_BUFFER_SIZE = 100

    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)


    model = create_model1(n_layers=0, width=5, drop_p=0.1, input_shape = 2*(12/n_prop_relevent ))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss=exponential_loss)
    history = model.fit(train_dataset, epochs=5, validation_data=(test_dataset), verbose=2)

    pred_params = model.predict(test_X)

    test_haz_rate = haz_rate[train_index:]
    test_haz_rate.shape = pred_params.shape

    conc_est = est_concordance(test_haz_rate, pred_params, s=200000)

    with open(work_dir + 'results/' + str(n_patient) + '_' + str(censored_p)+str(n_prop_relevent) + '_conc.csv', 'a',
              newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow([conc_est])

    errors = test_haz_rate-pred_params
    np.mean(errors)
    np.std(errors)

    fig = plt.scatter(pred_params, test_haz_rate, s = 1)
    plt.plot([0,250], [0,250], color='k', linestyle='-', linewidth=2)
    plt.savefig(work_dir+"figures/errors_"+str(n_patient)+'_'+str(censored_p)+'_'+str(n_prop_relevent)+'.png')
    plt.close()

for i in range(1,6):
    run_nn(n_patient = i*50000, censored_p=1)

for i in range(1,6):
    run_nn(n_patient = i*50000, censored_p=1)

for j in range(4):
    for i in range(1,6):
        run_nn(n_patient = 200000, censored_p=1, n_prop_relevent= 1/i)

for i in range(1,6):
    run_nn(n_patient = 50000, censored_p = 1, n_prop_relevent= i*(1/5))


#plt.hist(pred_params, bins = 100)
#plt.hist(test_haz_rate, bins = 100)
#plt.hist(errors, bins = 100)
#test_haz_rate[1:10]-y_pred[1:10]



