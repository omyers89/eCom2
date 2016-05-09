

# HW2 E-commerce Technion Spring 2016
# Written by: Omri Myers,        303083638
#             Amir Wolfensohn,   300339785

import csv
from sys import stdout
import numpy as np
from math import sqrt, fabs
from datetime import datetime




def make_dictionaries(data_file,  arcs_file,  test_file, validation_file=None, short = False):
    '''
    this function creates all the dictionaries and also calculate R_avg, Bu's, Bi's,
    it was originaly used to estimate the coeff therefore contain some modified un used
    parts.
    :param data_file: name of file contain the whole data set which is used to train the model row: [p_id,c_id,rank]
    :param arcs_file: name of file that contains the relations between products  row: [p1,p2]
    :param test_file: this csv is row [(p,c),-]
    :param validation_file: used to check the model accuracy used only when work in process row: [p_id,c_id,rank]
    :param short: for testing create only 1000/10 data dictionaries
    :return: dictionaries of the data used to predict the rankings
    '''
    with open(data_file, "r") as csv_file:
        reader = csv.DictReader(csv_file)
        customer_product_rank = {}
        product_customer_rank = {}
        customer_product_list = []
        #field_names = ['Product_ID', 'Customer_ID','Customer_rank' ]
        rank_sum = 0
        custom_rank_dict = {} #helper dictionary to create Bu's
        Bus_dict = {}
        product_rank_dict = {}
        Bis_dict = {}

        # every row in  customer_product_rank is (key:(customer, product), value:rank
        ai = 0
        for row in reader:
            r = int(row['Customer_rank'])
            c = row['Customer_ID']
            p = row['Product_ID']
            product_customer_rank[(p, c)] = r
            customer_product_rank[(c, p)] = r
            customer_product_list.append((row['Customer_ID'], row['Product_ID']))
            rank_sum += r
            #creating the Bu's
            if c in custom_rank_dict:
                custom_rank_dict[c] = np.append(custom_rank_dict[c], [r])
            else:
                custom_rank_dict[c] = np.array([r])
            #creating the Bi's
            if p in product_rank_dict:
                product_rank_dict[p] = np.append(product_rank_dict[p], [r])
            else:
                product_rank_dict[p] = np.array([r])
            ai+=1
            if short and ai > 1000:
                break
    if short:
        r_avg = float(rank_sum) / 1000.0
    else:
        r_avg = float(rank_sum) / float(len(product_customer_rank))

    for (ku, vlu) in custom_rank_dict.items():
        Bus_dict[ku] = np.average(vlu) - r_avg
    for (ki, vli) in product_rank_dict.items():
        Bis_dict[ki] = np.average(vli) - r_avg

    csv_file.close()

    #now making the dictionary of {[p,c] : pred} to predict
    test_dictionary = {}
    with open(test_file, "r") as csv_file_t:
        reader_t = csv.DictReader(csv_file_t)
        bi = 0
        for row in reader_t:
            c = row['Customer_ID']
            p = row['Product_ID']
            test_dictionary[p,c] = 0
            bi += 1
            if short and bi > 10:
                break

    csv_file_t.close()



    #for training purposes make another (p,c):r dictionary with real results
    if validation_file:
        res_dictionary = {}
        with open(validation_file, "r") as csv_file_r:
            reader_r = csv.DictReader(csv_file_r)
            ci = 0
            for row in reader_r:
                c = row['Customer_ID']
                p = row['Product_ID']
                r = int(row['Customer_rank'])
                res_dictionary[p, c] = r
                ci += 1
                if short and ci > 10:
                    break

        csv_file_t.close()
    else:
        res_dictionary = None


    #creat two dictionaries for relations between products
    neighbors_p1p2 = {}
    rev_neighbors_p2p1 = {}
    with open(arcs_file, "r") as csv_file_2:
        reader2 = csv.DictReader(csv_file_2)
        # remember : field_names = ['Product1_ID', 'Product2_ID']
        for row in reader2:
            if row['Product1_ID'] in neighbors_p1p2:
                neighbors_p1p2[row['Product1_ID']].append(row['Product2_ID'])
            else:
                neighbors_p1p2[row['Product1_ID']] = [row['Product2_ID']]
            if row['Product2_ID'] in rev_neighbors_p2p1:
                rev_neighbors_p2p1[row['Product2_ID']].append(row['Product1_ID'])
            else:
                rev_neighbors_p2p1[row['Product2_ID']] = [row['Product1_ID']]
    csv_file_2.close()

    return {'r_avg': r_avg, #average of all rankings
            'Bus_dict': Bus_dict, #average for every user
            'Bis_dict': Bis_dict,   #average for every product
            'customer_product_rank': customer_product_rank, #dictionary of [(c,p):r]
            'product_customer_rank': product_customer_rank, #dictionary of [(p,p):r]
            'test_dict': test_dictionary, #dictionary of [(p,c): 0] to put in the predictions
            'res_dict': res_dictionary, #dictionary of [(p,c): r] for validation
            'arcs': neighbors_p1p2, #dictionary of lists of similar products
            'rev_arcs': rev_neighbors_p2p1
            }



class arc_table():
    '''
    this class get the arc dictionaties and return for every combination user and product
    the similarity rank based on frequently bought together products
    '''
    def __init__(self, c_p_ranks, arcs_dict, rev_arcs_dict):
        self.ranks = c_p_ranks
        self.arcs = arcs_dict
        self.rev_arcs = rev_arcs_dict

    def get(self,u,i, rvg):
        r_sum = 0
        r_num = 0
        if i in self.rev_arcs:

            for p in self.rev_arcs[i]:
                if (u,p) in self.ranks:
                    r_sum += self.ranks[(u,p)]
                    r_num += 1

        if r_num == 0:
            r_res = rvg
        else:
            r_res = (float(r_sum) / r_num)
            # print "for u:{} and p:{} r_res is: {}".format(u, i, r_res)

        a_sum = 0
        a_num = 0
        if i in self.arcs:

            for pa in self.arcs[i]:
                if (u, pa) in self.ranks:
                    a_sum += self.ranks[(u, pa)]
                    a_num += 1
        if a_num == 0:
            a_res = rvg
        else:

            a_res = (float(a_sum) / a_num)
            # print "for u:{} and p:{} a_res is: {}".format(u, i,a_res )
        if not (r_res+a_res)/2.0-rvg == 0: print u,i,(r_res+a_res)/2.0-rvg
        return r_res,a_res, (r_res+a_res)/2.0-rvg


def base_line(dicts):

    #prediction constants, calculated by using linear regression:
    # the output of another program pre calculated[ 1.11085189  1.04354651  0.85812933  0.3491368 ]
    b0 = 1.11085189 #coef of r_avg
    b1 = 1.04354651 #coef of bu
    b2 = 0.85812933 #coef of bi
    b3 = 0.3491368 #coef of similarity


    r_avg = dicts['r_avg']
    bus = dicts['Bus_dict']
    bis = dicts['Bis_dict']

    c_p_rank = dicts['customer_product_rank']
    # r_roof = r_roof_table(dicts['r_avg'], dicts['Bus_dict'], dicts['Bis_dict'])
    arcs = dicts['arcs']
    rev_arcs = dicts['rev_arcs']
    arc_pred = arc_table(c_p_rank, arcs, rev_arcs)

    test_dict = dicts['test_dict']
    iit = 0
    for (p_id, c_id) in test_dict:
        iit += 1
        r = str(iit) + "\r"
        stdout.write(r)
        bu = 0
        bi = 0
        if c_id in bus:
            bu = bus[c_id]
        if p_id in bis:
            bi = bis[p_id]
        simr_pred, sima_pred, avg_sim_norm = arc_pred.get(c_id, p_id, r_avg)
        test_dict[p_id, c_id] = b0*r_avg + b1*bu + b2*bi + b3*avg_sim_norm

    return test_dict


def print_results(p_c_predictions, target_file):
    with open(target_file, 'w') as write_file:
        writer = csv.writer(write_file, lineterminator='\n')
        fieldnames2 = ["Product_ID", "Customer_ID", "Customer_rank"]
        writer.writerow(fieldnames2)
        # rmse = 0
        row_count = 0
        for (p,c),r in p_c_predictions.items():

            if round(r) > 5:
                t_pred = 5
            elif round(r) < 1:
                t_pred = 1
            else:
                t_pred = round(r)

            writer.writerow([p, c, int(t_pred)])
            row_count +=1

    write_file.close()
    return row_count


def run_prediction(dicts, target_file):
    p_c_predictions = base_line(dicts)
    print_results(p_c_predictions, target_file)



def calc_rmse(my_pred, real_labels):
    '''
    given two dictionaries calculate the rmse and evaluate the prediction
    :param my_pred: name of csv file contains the model predictions
    :param real_labels: name of csv file contains the real labels
    :return: RMSE and list of the miss classified entries
    '''

    test_size = len(real_labels)
    diff_sum = 0
    dbg_list = []
    for (p,c),v in real_labels.items():
        r_diff = v-my_pred[(p,c)]
        if r_diff > 0:
            dbg_list.append({'key': (p, c),
                             'r_val': v,
                             'my_val': my_pred[(p, c)]})

        diff_sum += r_diff**2
    rmses = float(diff_sum) / test_size
    return sqrt(rmses), dbg_list

def test_results(pred_file, real_res):
    '''
    given two csv files with row [(p,c) rank] calculates the rmse and evaluate the prediction
    :param pred_file: name of csv file contains the model predictions
    :param real_res: name of csv file contains the real labels
    :return: RMSE and list of the miss classified entries
    '''
    pred_dict = {}
    with open(pred_file,'r') as predictions:
        reader_p = csv.DictReader(predictions)
        for row in reader_p:
            r = int(row['Customer_rank'])
            c = row['Customer_ID']
            p = row['Product_ID']
            pred_dict[(p, c)] = r
    predictions.close()
    real_dict = {}
    with open(real_res, 'r') as labels:
        reader_l = csv.DictReader(labels)
        for row in reader_l:
            r = int(row['Customer_rank'])
            c = row['Customer_ID']
            p = row['Product_ID']
            real_dict[(p, c)] = r
    labels.close()
    rmse, db_list = calc_rmse(pred_dict, real_dict)
    print "rmse is:", rmse
    return db_list



if __name__ == '__main__':

    t1 = datetime.now()

    '''get the input from file "P_C_matrix.csv" and "Network_arcs.csv"
    make the dictionaries
    call the regression with the parameters
    create the "EX2.csv" file'''

    data_file = "P_C_matrix.csv"
    arcs_file = "Network_arcs.csv"
    results_file = "results.csv"
    target_file_name = "EX2.csv"

    #for debug:
    # pref = raw_input("choose long or short: (L or S)")
    # if pref == "L":
    #     short = False
    # elif pref == "S":
    #     short = True

    all_dicts = make_dictionaries(data_file,  arcs_file,  results_file, None, short=False)
    run_prediction(all_dicts, target_file_name)

    real_res_file = ""
    if not real_res_file == "":
        test_results(target_file_name, real_res_file)

    t2 = datetime.now()
    print (t2 - t1)
