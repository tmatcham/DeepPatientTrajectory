# Inputs - number of output variables, number of input variables????

#Plan: create integrated hazard rate for output based on input variables

#Start with random information - categorical and numerical

import numpy as np
import random

data_path = 'D:/PycharmProjects/EHR_survival_prediction/single_risk_censored/data/'

def generate_data(n_prop_relevent = 0.5 ,n_patient = 2500,n_cat_var = 20, n_cts_var = 20, censored_p = 0.5):
    cat_vars = np.zeros((n_patient, n_cat_var))
    for i in range(n_cat_var):
        cat_vars[:,i] = np.random.binomial(1,0.5, size = n_patient)
    cat_vars -= 0.5
    cts_vars = np.random.normal(0,1, size = n_patient*n_cts_var)
    cts_vars.shape = (n_patient, n_cts_var)
    all_vars = np.concatenate((cat_vars, cts_vars), axis = 1 )

    #create output variables - create hazard rate and then produce events accordingly
    #Decide which fraction of variables will be influential
    n_relevent_cat = int(n_cat_var*n_prop_relevent)
    n_relevent_cts = int(n_cts_var*n_prop_relevent)

    #select relevent variables
    relevent_cat_ind = random.sample(list(range(n_cat_var)), n_relevent_cat)
    relevent_cts_ind = random.sample(list(range(n_cat_var,n_cat_var+n_cts_var)), n_relevent_cts)
    ind = relevent_cat_ind.copy()
    ind.extend(relevent_cts_ind)

    #Make the hazard rate a linear function of the relevent variables - make all coefficient positive
    Beta = np.random.uniform(5,10, n_relevent_cat+n_relevent_cts)

    vars_rel = all_vars[:, ind]

    haz_rate = np.matmul(vars_rel, Beta)
    haz_rate = haz_rate+ 1-(min(haz_rate))
    inverse_prob_quants = np.random.uniform(0,1,n_patient)
    t = -np.log(1-inverse_prob_quants)/haz_rate
    C = np.quantile(t,censored_p)
    obs_ind = (t>=C)
    t[obs_ind] = C


    #save everything

    np.savetxt(data_path + "Beta"+str(n_patient)+'_'+str(censored_p)+'_'+str(n_prop_relevent)+".csv", Beta,delimiter=",")
    np.savetxt(data_path + "vars_all"+str(n_patient)+'_'+str(censored_p)+'_'+str(n_prop_relevent)+".csv", all_vars, delimiter=",")
    np.savetxt(data_path + "ind_rel"+str(n_patient)+'_'+str(censored_p)+'_'+str(n_prop_relevent)+".csv", ind, delimiter=",")
    np.savetxt(data_path + "times"+str(n_patient)+'_'+str(censored_p)+'_'+str(n_prop_relevent)+".csv", t,delimiter=",")
    np.savetxt(data_path + "obs_ind"+str(n_patient)+'_'+str(censored_p)+'_'+str(n_prop_relevent)+".csv", obs_ind,delimiter=",")
    np.savetxt(data_path + "haz_rate"+str(n_patient)+'_'+str(censored_p)+'_'+str(n_prop_relevent)+".csv", haz_rate, delimiter=",")
    print('Done')
    return None

for i in range(1,6):
    generate_data(n_patient=50000*i, n_cat_var=20, n_cts_var=20, censored_p=1)

for i in range(5):
    generate_data(n_patient=200000, n_cat_var=20, n_cts_var=20, censored_p=0.2*(i))

for i in range(1,6):
    generate_data(n_patient = i*50000, n_cat_var= 20, n_cts_var=20, censored_p = 1, n_prop_relevent= 1/i)


for i in range(1,6):
    generate_data(n_patient = 50000, n_cat_var= int(12/(i/5)), n_cts_var=int(12/(i/5)), censored_p = 1, n_prop_relevent= i*(1/5))

