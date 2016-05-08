# from sklearn.ensemble import BaggingClassifier
import csv
# import sklearn.linear_model
from math import sqrt
import numpy as np
from sys import stdout
def calc_rmse_w(fbgcm, validation_set,validation_set_labels ):
    pred_res = fbgcm.predict(validation_set)
    with open('EX2.csv', 'w') as write_file:
        writer = csv.writer(write_file, lineterminator='\n')
        fieldnames2 = ["Proudct_ID", "Customer_ID", "Customer_rank"]
        writer.writerow(fieldnames2)
        rmse = 0
        for l, p in zip(validation_set_labels, pred_res):
            # print  l[1], l[0], p
            if round(p) > 5:
                t_pred = 5
            elif round(p) < 1:
                t_pred = 1
            else:
                t_pred = round(p)
            r_d = l - t_pred
            rmse += pow(r_d, 2)
            writer.writerow(['p', 'c', t_pred])
    write_file.close
    # print pred_res[:10]

    norm_rmse = float(rmse) / len(validation_set_labels)
    print "rmse is:"
    print sqrt(norm_rmse)

def calc_rmse(fbgcm, validation_set,validation_set_labels ):
    pred_res = fbgcm.predict(validation_set)

    rmse = 0
    for l, p in zip(validation_set_labels, pred_res):
        # print  l[1], l[0], p
        if round(p) > 5:
            t_pred = 5
        elif round(p) < 1:
            t_pred = 1
        else:
            t_pred = round(p)
        r_d = l - t_pred
        rmse += pow(r_d, 2)
    norm_rmse = float(rmse) / len(validation_set_labels)
    return sqrt(norm_rmse)


class linear_solver():

    def __init__(self,rvg,buv,biv,aa,ar):
        self.c_rvg = rvg
        self.c_buv = buv
        self.c_biv = biv
        self.c_aa = aa
        self.c_ar = ar
        self.coef_vec = np.array([self.c_rvg,self.c_buv,self.c_biv,self.c_aa,self.c_ar])

        self.coef_dict = {'rvg': self.c_rvg,
                          'buv': self.c_buv,
                          'biv': self.c_biv,
                           'aa': self.c_aa,
                           'ar': self.c_ar}

    def predict(self, data):
        predictions = np.zeros(len(data))
        for i, d_vec in enumerate(data):
            np_dvec = (np.array(d_vec)).transpose()
            predictions[i] = np.dot(self.coef_vec, np_dvec)
        return predictions

    # def __str__(self):
    #     return self.coef_vec






def run_linear_grid(model_name, training_set, train_set_labels, validation_set=None, validation_set_labels=None , facc=False):
    print "*********fiting model -", model_name,"**************"
    coef_hi_values = [x/6.2 for x in range(9,3,-1)]
    coef_low_values = [y/6.2 for y in range(5,0,-1)]

    best_rmse = float('inf')
    iit = 0
    for rvg in coef_hi_values:
        for buv in coef_hi_values:
            for biv in coef_hi_values:
                for aa in coef_low_values:
                    for ar in coef_low_values:
                        iit += 1
                        r = str(iit) + "\r"
                        stdout.write(r)
                        solver = linear_solver(rvg,buv,biv,aa,ar)
                        train_rmse = calc_rmse(solver, training_set, train_set_labels)
                        valid_rmse = calc_rmse(solver, validation_set, validation_set_labels)
                        if train_rmse < best_rmse:
                            best_solver = solver
                            best_rmse = train_rmse
                            print 'new solver found.'
                            print 'best train_rmse is:',best_rmse
                            print "best valid_rmse is:", valid_rmse
                            print "and coeffs are:"
                            print best_solver.coef_dict

    print "coefs of the model are:"
    print best_solver.coef_dict
    return best_solver,best_rmse

def run_linear_grid_rig(model_name, base_coefs, training_set, train_set_labels, validation_set=None, validation_set_labels=None , facc=False):
    print "*********fiting rig model -", model_name,"**************"
    new_coefs = {}
    for c,v in base_coefs.items():
        e_c = v
        new_coefs[c] = [(e_c + x/16.0) for x in range(-4,+4)]

   #fintuning the coeffs
    best_rmse = float('inf')
    for rvg in new_coefs['rvg']:
        for buv in new_coefs['buv']:
            for biv in new_coefs['biv']:
                for aa in [0.15]: #new_coefs['aa']:
                    for ar in [0.15]: # new_coefs['ar']:
                        solver = linear_solver(rvg,buv,biv,aa,ar)
                        train_rmse = calc_rmse(solver, training_set, train_set_labels)
                        valid_rmse = calc_rmse(solver, validation_set, validation_set_labels)
                        if train_rmse < best_rmse:
                            best_solver = solver
                            best_rmse = train_rmse
                            print 'new solver found.'
                            print 'best train_rmse is:', best_rmse
                            print "best valid_rmse is:", valid_rmse
                            print "and coeffs are:"
                            print best_solver.coef_dict

    print "coefs of the model are:"
    print best_solver.coef_dict
    return best_solver,best_rmse







