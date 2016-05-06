
import csv
from sys import stdout
import numpy as np
from math import sqrt, fabs
from datetime import datetime

def make_dictionaries(data_file, test_file, res_file = None, short = False):
    '''
    this function creates all the dictionaries and also calculate R_avg, Bu's, Bi's
    :param product_customer_rank:
    :param customer_product_rank:
    :param customer_product_list:
    :param product_neighbors:
    :return: R_avg, Bu's, Bi's
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


    if res_file:
        res_dictionary = {}
        with open(res_file, "r") as csv_file_r:
            reader_r = csv.DictReader(csv_file_r)
            ci = 0
            for row in reader_r:
                c = row['Customer_ID']
                p = row['Product_ID']
                r = row['Customer_rank']
                res_dictionary[p, c] = r
                ci += 1
                if short and ci > 10:
                    break

        csv_file_t.close()
    else:
        res_dictionary = None



    # with open("Network_arcs.csv", "r") as csv_file_2:
    #     reader2 = csv.DictReader(csv_file_2)
    #     # remember : field_names = ['Product1_ID', 'Product2_ID']
    #     for row in reader2:
    #         if row['Product1_ID'] in product_neighbors:
    #             product_neighbors[row['Product1_ID']].append(row['Product2_ID'])
    #         else:
    #             product_neighbors[row['Product1_ID']] = [row['Product2_ID']]
    # csv_file_2.close()

    return {'r_avg': r_avg,
            'Bus_dict': Bus_dict,
            'Bis_dict': Bis_dict,
            'customer_product_rank': customer_product_rank,
            'test_dict': test_dictionary,
            'res_dict': res_dictionary
            }



# r_orig = r_orig_table(customer_product_rank)
# r_roof = r_roof_table(rvg, bus, bis)
# r_tilda = r_tilda_table(r_orig, r_roof)
# D = d_table(r_tilda)
# r_roof_new = r_roof_new_table(r_roof, r_tilda, D)


class r_roof_table():
    def __init__(self, rvg, bu_dict, bi_dict):
        self.r_avg = rvg
        self.bus = bu_dict
        self.bis = bi_dict

    def get_bu(self, user_id):
        if user_id in self.bus:
            return self.bus[user_id]
        else:
            return 0

    def get_bi(self, product_id):
        if product_id in self.bis:
            return self.bis[product_id]
        else:
            return 0

    def get(self,u,i):
        val_ui = self.r_avg + self.get_bu(u) + self.get_bi(i)
        return val_ui

class r_tilda_table():
    def __init__(self, r_orig, r_roof):
        self.r_table = r_orig
        self.roof = r_roof
    def get_r(self,user_id, product_id):
        if (user_id, product_id) in self.r_table:
            return self.r_table[(user_id, product_id)]
        else:
            return 0

    def get_users(self):
        user_list = [u for u in self.roof.bus]
        return user_list

    def get_products(self):
        products = [p for p in self.roof.bis]
        return products

    def get(self, u,i):
        r_ui = self.get_r(u,i)
        if r_ui > 0:
            return r_ui - self.roof.get(u,i)
        else:
            return 0.0

    def get_prod_np(self, product_id):
        pnp = np.array([]) #np array of the tilda value of all users in r_tilda[u][product_id]
        for uu in self.get_users():
            pnp = np.append(pnp, [self.get(uu,product_id)])

        return pnp


class d_table():
    def __init__(self, r_tilda):
        self.tildas = r_tilda

    def get_users(self):
        user_list = self.tildas.get_users()
        return user_list

    def l_most_similar(self, product_id, l=7):
        sim_list = []
        for p in self.tildas.get_products():
            if not p == product_id:
                sim_list.append((p , self.get(product_id,p)))

        sim_list.sort(key=lambda x: x[1])
        return sim_list[-l:]

    def get(self,i,j):
        # users = self.get_users()

        ris = self.tildas.get_prod_np(i) # all Rui
        rjs = self.tildas.get_prod_np(j) # all Ruj
        numerator = np.dot(ris, np.transpose(rjs))  # sum of Rui*Ruj
        denumerator = sqrt(sum(np.power(ris, 2))) * sqrt(sum(np.power(rjs, 2)))
        if denumerator == 0:
            return -1
        else:
            return float(numerator)/float(denumerator)



class r_roof_new():
    def __init__(self, roof_table, dr_table, tilda_table, n_sim=7):
        self.roofs = roof_table
        self.ds = dr_table
        self.tildas = tilda_table
        self.l = n_sim

    def get(self,u,i):
        roof_ui = self.roofs.get(u,i)
        d_similars = self.ds.l_most_similar(i , self.l)
        numer = 0
        denumer = 0
        print "in get r_roof_new"
        for (sim_product, d_val) in d_similars:
            d_ij = self.ds.get(i,sim_product)
            r_uj = self.tildas.get(u, sim_product)
            numer += d_ij * r_uj
            denumer += fabs(d_ij)
            print sim_product
        if denumer == 0:
            return roof_ui
        else:
            diff_ui = float(numer) / float(denumer)
            return roof_ui + diff_ui

def print_tables(t , name):
    print "this is:", name
    for uuii in range(1, 5):
        for ppjj in range(1, 5):
            stdout.write(str(t.get(uuii, ppjj))[:4] +", \t\t")
        stdout.write('\n')

def test_tables(traning_data, test_set):
    # c_p_rank = {(1,1):5,          (1,3):5, (1,4):2,
    #                      (2,2):4, (2,3):4, (2,4):4,
    #             (3,1):2, (3,2):4, (3,3):3, (3,4):3}
    # bus = {1:0.4, 2:0.4, 3:-0.6}
    # bis = {1:-0.1, 2:0.4, 3:0.4, 4:-0.6}
    # rvg = 3.6
    product_customer_rank = {}
    # {(P_i,C_j):rank , (P_m,C_n):rank , .....}

    customer_product_rank = {}
    # {(C_i,P_j):rank , (C_m,P_n):rank , .....}

    customer_product_list = []
    # [(C_i,P_j), (C_m,P_n), .....]

    product_neighbors = {}
    # {product1:[product1_neighbor1 , product1_neighbor2, ...] , product2:[product2_neighbor1 , product2_neighbor2, ...], ... }


    rvg, bus, bis = make_dictionaries(product_customer_rank,
                                      customer_product_rank,
                                      customer_product_list,
                                      product_neighbors)

    r_orig = customer_product_rank
    r_roof = r_roof_table(rvg, bus, bis)
    r_tilda = r_tilda_table(r_orig, r_roof)
    D = d_table(r_tilda)
    r_r_new = r_roof_new(r_roof, D, r_tilda, 2)

    print_tables(r_roof, "r_roof")
    print_tables(r_tilda, "r_tilda")
    print_tables(D, "D")
    print_tables(r_r_new, "r_roof_new")

def csv_test(d_file, t_file, r_file, short = False):
    product_customer_rank = {}
    # {(P_i,C_j):rank , (P_m,C_n):rank , .....}

    customer_product_rank = {}
    # {(C_i,P_j):rank , (C_m,P_n):rank , .....}

    customer_product_list = []
    # [(C_i,P_j), (C_m,P_n), .....]

    product_neighbors = {}
    # {product1:[product1_neighbor1 , product1_neighbor2, ...] , product2:[product2_neighbor1 , product2_neighbor2, ...], ... }




    dicts = make_dictionaries(d_file,
                              t_file,
                              r_file, short)

    print 'make_dicts: Done'
    r_orig = dicts['customer_product_rank']

    r_roof = r_roof_table(dicts['r_avg'], dicts['Bus_dict'], dicts['Bis_dict'])
    print 'rroof : Done'
    r_tilda = r_tilda_table(r_orig, r_roof)
    D = d_table(r_tilda)
    print 'dd : Done'
    r_r_new = r_roof_new(r_roof, D, r_tilda, 2)

    test_dict = dicts['test_dict']
    iit=0
    for (p_id, c_id) in test_dict:
        iit += 1
        r = str(iit) + "\r"
        stdout.write(r)
        test_dict[p_id, c_id] = r_r_new.get(c_id, p_id)


    res_dict = dicts['res_dict']

    rmse = 0
    for t in test_dict:
        r_d = test_dict[t] - res_dict[t]
        rmse += pow(r_d, 2)
    norm_rmse = float(rmse) / float(len(test_dict))
    print "rmse is:"
    print sqrt(norm_rmse)



if __name__ == '__main__':
    # print "try tables"
    # test_tables()
    t1 = datetime.now()
    csv_test('15-fold_0_training.csv', '15-fold_0_test.csv', '15-fold_0_test_labeled.csv', short = True)
    t2 = datetime.now()
    print (t2 - t1)
