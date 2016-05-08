import csv
from random import random, sample

def get_data(data_file, short = False):
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
        product_customer_rank = {}
        product_customer_list = []
        #field_names = ['Product_ID', 'Customer_ID','Customer_rank' ]


        # every row in  customer_product_rank is (key:(customer, product), value:rank
        ai = 0
        for row in reader:
            r = int(row['Customer_rank'])
            c = row['Customer_ID']
            p = row['Product_ID']
            product_customer_rank[(p, c)] = r

            product_customer_list.append(((p, c), False))


            ai+=1
            if short and ai > 1000:
                break

    csv_file.close()

    return product_customer_rank, product_customer_list







def make_csvs(training=0.9, testinng=0.1, num_of_sets=10, short=False):
    p_c_data, p_c_list = get_data('P_C_matrix', short)
    n_data = len(p_c_list)

    for i in range(num_of_sets):
        #get testing_set
        p_c_dict
        n_test = (round(testinng*n_data)+1)
        sam_indexes = sample(xrange(n_data), n_test)
        for



    with open('EX2.csv', 'w') as write_file:
        writer = csv.writer(write_file, lineterminator='\n')
        fieldnames2 = ["Proudct_ID", "Customer_ID", "Customer_rank"]
        writer.writerow(fieldnames2)
        rmse = 0
        for l, p in zip(validation_set_labels, pred_res):
            # print  l[1], l[0], p
            r_d = l[1] - round(p)
            rmse += pow(r_d, 2)
            if round(p) > 5 or round(p) < 1: print round(p), "************** "
            writer.writerow([l[0][0], l[0][1], round(p)])
    write_file.close




if __name__ == '__main__':
    v = True
    make_csvs(training=0.9, testinng=0.1, num_of_sets=10, short=v)
