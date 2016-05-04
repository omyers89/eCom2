
import csv

import numpy as np
from math import sqrt, fabs

def make_dictionaries(product_customer_rank, customer_product_rank,customer_product_list,product_neighbors):
    '''
    this function creates all the dictionaries and also calculate R_avg, Bu's, Bi's
    :param product_customer_rank:
    :param customer_product_rank:
    :param customer_product_list:
    :param product_neighbors:
    :return: R_avg, Bu's, Bi's
    '''
    with open("P_C_matrix.csv", "r") as csv_file:
        reader = csv.DictReader(csv_file)
        #field_names = ['Product_ID', 'Customer_ID','Customer_rank' ]
        rank_sum = 0
        custom_rank_dict = {} #helper dictionary to create Bu's
        Bus_dict = {}
        product_rank_dict = {}
        Bis_dict = {}

        # every row in  customer_product_rank is (key:(customer, product), value:rank
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
                custom_rank_dict[c] = np.append(custom_rank_dict[c], r)
            else:
                custom_rank_dict[c] = np.array([r])
            #creating the Bi's
            if p in product_rank_dict:
                product_rank_dict[p] = np.append(product_rank_dict[p], r)
            else:
                product_rank_dict[p] = np.array([r])
    r_avg = float(rank_sum) / float(len(product_customer_rank))
    for (ku, vlu) in custom_rank_dict.items():
        Bus_dict[ku] = np.average(vlu) - r_avg
    for (ki, vli) in product_rank_dict.items():
        Bis_dict[ki] = np.average(vli) - r_avg

    csv_file.close()

    with open("Network_arcs.csv", "r") as csv_file_2:
        reader2 = csv.DictReader(csv_file_2)
        # remember : field_names = ['Product1_ID', 'Product2_ID']
        for row in reader2:
            if row['Product1_ID'] in product_neighbors:
                product_neighbors[row['Product1_ID']].append(row['Product2_ID'])
            else:
                product_neighbors[row['Product1_ID']] = [row['Product2_ID']]
    csv_file_2.close()

    return r_avg, Bus_dict, Bis_dict



# r_orig = r_orig_table(customer_product_rank)
# r_roof = r_roof_table(rvg, bus, bis)
# r_tilda = r_tilda_table(r_orig, r_roof)
# D = d_table(r_tilda)
# r_roof_new = r_roof_new_table(r_roof, r_tilda, D)


class r_roof_table():
    def __init__(self, rvg, bu_dict,bi_dict):
        r_avg = rvg
        bus = bu_dict
        bis = bi_dict

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

    def get_val_ui(self,u,i):
        val_ui = self.r_avg + self.get_bu(u) + self.get_bi(i)
        return val_ui

class r_tilda_table():
    def __init__(self, r_orig, r_roof):
        r_table = r_orig
        roof = r_roof
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

    def get_tilda_ui(self, u,i):
        r_ui = self.get_r(u,i)
        if r_ui > 0:
            return r_ui - self.roof.get_val_ui(u,i)
        else:
            return 0.0

    def get_prod_np(self, product_id):
        pnp = np.array() #np array of the tilda value of all users in r_tilda[u][product_id]
        for u in self.get_users():
            np.append(pnp, self.get_tilda_ui(u,product_id))

        return pnp


class d_table():
    def __init__(self, r_tilda):
        tildas = r_tilda

    def get_users(self):
        user_list = self.tildas.get_users()
        return user_list

    def l_most_similar(self, product_id, l=7):
        sim_list = []
        for p in self.tildas.get_products():
            if not p == product_id:
                sim_list.append(p , self.get_d_ij(product_id,p))

        sim_list.sort(key=lambda x: x[1])
        return sim_list[-l:]

    def get_d_ij(self,i,j):
        users = self.get_users()

        ris = self.tildas.get_prod_np(i) # all Rui
        rjs = self.tildas.get_prod_np(j) # all Ruj
        numerator = np.dot(ris, np.transpose(rjs))  # sum of Rui*Ruj
        denumerator = sqrt(sum(ris^2)) * sqrt(sum(rjs^2))
        return float(numerator)/float(denumerator)



class r_roof_new():
    def __init__(self, roof_table, d_table, tilda_table):
        roofs = roof_table
        ds = d_table
        tildas = tilda_table

    def get_new_ui(self,u,i):
        roof_ui = self.roofs.get_val_ui(u,i)
        d_similars = self.ds.l_most_similar(i)
        numer = 0
        denumer = 0
        for (sim_product, d_val) in d_similars:
            d_ij = self.ds.get_d_ij(i,sim_product)
            r_uj = self.tildas.get_tilda_ui(u, sim_product)
            numer +=  d_ij * r_uj
            denumer += fabs(d_ij)
        diff_ui = float(numer) / float(denumer)
        return roof_ui + diff_ui



if __name__ == '__main__':
    print "try tables"
    # test_tables()
