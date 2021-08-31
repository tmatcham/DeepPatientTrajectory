# Inputs - number of output variables, number of input variables????

#Plan: create integrated hazard rate for output based on input variables

#Start with random information - categorical and numerical

import numpy as np
import random

def generate_data(data_path, n_patient = 2500,n_cat_var = 20, n_cts_var = 20,
                  censored_p = 0.8, n_prop_relevent=0.5, n_risks = 2):
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
    for r in range(n_risks):
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

        t.shape = (n_patient, 1)
        Beta.shape = (n_relevent_cat + n_relevent_cts, 1)
        ind = np.asarray(ind)
        ind.shape = (n_relevent_cat + n_relevent_cts, 1)
        haz_rate.shape = (n_patient, 1)
        obs_ind.shape = (n_patient, 1)

        if r == 0:
            times = t
            Beta_array = Beta
            ind_array = ind
            haz_rate_array = haz_rate
            obs_ind_array = obs_ind
        else:
            times = np.concatenate((times, t), axis=1)
            Beta_array = np.concatenate((Beta_array, Beta), axis=1)
            ind_array = np.concatenate((ind_array, ind), axis=1)
            haz_rate_array = np.concatenate((haz_rate_array, haz_rate), axis=1)
            obs_ind_array = np.concatenate((obs_ind_array,obs_ind), axis=1)

    t_obs = np.min(times, axis=1)
    r_obs = np.argmin(times, axis=1)
    r_obs += 1
    r_obs[np.prod(obs_ind_array, axis = 1)==1] = 0

    np.savetxt(data_path + "vars_all.csv", all_vars, delimiter=",")
    np.savetxt(data_path + "t_obs.csv", t_obs, delimiter=",")
    np.savetxt(data_path + "r_obs.csv", r_obs, delimiter=",")
    np.savetxt(data_path + "Beta_array.csv", Beta_array, delimiter=",")
    np.savetxt(data_path + "ind_array.csv", ind_array, delimiter=",")
    np.savetxt(data_path + "haz_rate_array.csv", haz_rate_array, delimiter=",")
    np.savetxt(data_path + "obs_ind_array.csv", obs_ind_array, delimiter=",")
    return None



