
from tables import make_dictionaries

from datetime import datetime,time, timedelta






#dont use these two:\/

# def make_Bus(customer_product_rank):
#     custom_rank_dict = {}
#     Bus_dict = {}
#     #every row in  customer_product_rank is (key:(customer, product), value:rank
#     for ((c,p),v) in customer_product_rank.items():
#         if c in custom_rank_dict:
#             custom_rank_dict[c] = np.append(custom_rank_dict[c], v)
#         else:
#             custom_rank_dict[c] = np.array([v])
#             # custom_rank_dict[c].append(v)
#     for (kk, vl) in custom_rank_dict.items():
#         Bus_dict[kk] = np.average(vl)
#     return Bus_dict
#
# def make_Bis(product_customer_rank):
#     product_rank_dict = {}
#     Bis_dict = {}
#     #every row in  customer_product_rank is (key:(customer, product), value:rank
#     for ((p,c),v) in product_customer_rank.items():
#         if p in product_rank_dict:
#             product_rank_dict[p] = np.append(product_rank_dict[p], v)
#         else:
#             product_rank_dict[p] = np.array([v])
#             # custom_rank_dict[c].append(v)
#     for (kk, vl) in product_rank_dict.items():
#         Bis_dict[kk] = (np.average(vl),vl.size)
#
#
#     return Bis_dict


def make_r_orig(customer_product_rank):
    customer_dict = {}
    for (c,p),v in customer_product_rank.items():
        if c in customer_dict:
            customer_dict[c][p] = v
        else:
            customer_dict[c] = {}
            customer_dict[c][p] = v
    return customer_dict


#
def make_r_roof(rvg, bus, bis):
    customer_dict = {}
    for u,bu in bus.items():
        customer_dict[u]={}
        for i, bi in bis.items():
            customer_dict[u][i] = rvg + bu + bi

    return customer_dict


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




    # for i, rec in enumerate(r_roof.items()):
    #     print rec[1].items()[:10]
    #     if i > 20: break

    print "we have {} users and {} products".format(len(bus), len(bis))

    #
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
    # print "this is 20 first bus"
    #
    # for i, rec in enumerate(bus.items()):
    #     print rec
    #     if i > 20: break
    #
    # print "this is 20 first bis"
    #
    # for i, rec in enumerate(bis.items()):
    #     print rec
    #     if i > 20: break

    t2 = datetime.now()
    print (t2 - t1)
