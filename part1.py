
import csv



def make_dictionaries(product_customer_rank, customer_product_rank,customer_product_list,product_neighbors):
    with open("P_C_matrix.csv", "r") as csv_file:
        reader = csv.DictReader(csv_file)
        # remember : field_names = ['Product_ID', 'Customer_ID','Customer_rank' ]
        for row in reader:
            product_customer_rank[(row['Product_ID'], row['Customer_ID'])] = int(row['Customer_rank'])
            customer_product_rank[(row['Customer_ID'], row['Product_ID'])] = int(row['Customer_rank'])
            customer_product_list.append((row['Customer_ID'], row['Product_ID']))
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


if __name__ == '__main__':
    product_customer_rank = {}
    # {(P_i,C_j):rank , (P_m,C_n):rank , .....}

    customer_product_rank = {}
    # {(C_i,P_j):rank , (C_m,P_n):rank , .....}

    customer_product_list = []
    # [(C_i,P_j), (C_m,P_n), .....]

    product_neighbors = {}
    # {product1:[product1_neighbor1 , product1_neighbor2, ...] , product2:[product2_neighbor1 , product2_neighbor2, ...], ... }

    make_dictionaries(product_customer_rank, customer_product_rank,customer_product_list,product_neighbors)

    for i,d in enumerate(product_customer_rank):
        cmr = d[1]
        prd = d[0]
        print "pcr:"
        print d, product_customer_rank[d]
        print "cpr:"
        print (cmr,prd), customer_product_rank[(cmr,prd)]
        print "cpl:"
        print customer_product_list[i]
        print "pn:"
        if prd in product_neighbors: print product_neighbors[prd]
        else: print "none"


        if i == 3 : break