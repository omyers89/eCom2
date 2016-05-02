
import csv

import numpy as np
from datetime import datetime,time, timedelta



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
    for (ku, vlu) in custom_rank_dict.items():
        Bus_dict[ku] = (np.average(vlu), vlu.size)
    for (ki, vli) in product_rank_dict.items():
        Bis_dict[ki] = (np.average(vli), vli.size)

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
    r_avg = float(rank_sum)/float(len(product_customer_rank))
    return r_avg, Bus_dict, Bis_dict



#dont use these two:\/

def make_Bus(customer_product_rank):
    custom_rank_dict = {}
    Bus_dict = {}
    #every row in  customer_product_rank is (key:(customer, product), value:rank
    for ((c,p),v) in customer_product_rank.items():
        if c in custom_rank_dict:
            custom_rank_dict[c] = np.append(custom_rank_dict[c], v)
        else:
            custom_rank_dict[c] = np.array([v])
            # custom_rank_dict[c].append(v)
    for (kk, vl) in custom_rank_dict.items():
        Bus_dict[kk] = np.average(vl)
    return Bus_dict

def make_Bis(product_customer_rank):
    product_rank_dict = {}
    Bis_dict = {}
    #every row in  customer_product_rank is (key:(customer, product), value:rank
    for ((p,c),v) in product_customer_rank.items():
        if p in product_rank_dict:
            product_rank_dict[p] = np.append(product_rank_dict[p], v)
        else:
            product_rank_dict[p] = np.array([v])
            # custom_rank_dict[c].append(v)
    for (kk, vl) in product_rank_dict.items():
        Bis_dict[kk] = (np.average(vl),vl.size)


    return Bis_dict

if __name__ == '__main__':
    t1 = datetime.now()
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

    # for i,d in enumerate(product_customer_rank):
    #     cmr = d[1]
    #     prd = d[0]
    #     print "pcr:"
    #     print d, product_customer_rank[d]
    #     print "cpr:"
    #     print (cmr,prd), customer_product_rank[(cmr,prd)]
    #     print "cpl:"
    #     print customer_product_list[i]
    #     print "pn:"
    #     if prd in product_neighbors: print product_neighbors[prd]
    #     else: print "none"


        # if i == 3 : break
    print "this is 20 first bus"

    for i, rec in enumerate(bus.items()):
        print rec
        if i > 20: break

    print "this is 20 first bis"

    for i, rec in enumerate(bis.items()):
        print rec
        if i > 20: break

    t2 = datetime.now()
    print (t2 - t1)
