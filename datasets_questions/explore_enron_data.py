#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000

"""

import pickle
from pprint import pprint

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
print "Number of person:{}".format(len(enron_data))

for k, v in enron_data.items():
    # pprint(k)
    # print "Number of feature per person:{}".format(len(v))
    pprint(v)
    break
    pass

num_poi = 0
for k in enron_data:
    if enron_data[k]['poi'] == 1:
        num_poi += 1

print "Number of POI:{}".format(num_poi)

###
total_stock_value = enron_data['PRENTICE JAMES']['total_stock_value']
print "Total stock value is: %s" % total_stock_value
###
email_num = enron_data['Colwell Wesley'.upper()]['from_this_person_to_poi']
print "Total email num from Colwell Wesley to POI is: %s" % email_num

num_total_payment_NaN = 0
for k, v in enron_data.items():
    if v['total_payments'] == 'NaN':
        num_total_payment_NaN += 1
print "Percentage whose total payment is NaN: %s" % (num_total_payment_NaN * 1.0 / len(enron_data))
