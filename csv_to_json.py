import pandas as pd
import csv
import json

test = '/COVID_8TB/vlp_results/csv_files/medvill_seed777_label.csv'
gt = '/COVID_8TB/vlp_results/csv_files/medvill_seed777_pred.csv'

f = open(test, 'r')
rdr = csv.reader(f)
test_csv = []
for line in rdr:
    test_csv.append(line[0])
f.close()

f = open(gt, 'r')
rdr = csv.reader(f)
gt_csv = []
for line in rdr:
    gt_csv.append(line[0])
f.close()

json_file = {}

for i, (t, g) in enumerate(zip(test_csv, gt_csv)):
    json_file[i] = {'predicted': t, 'caption': g}

with open('/COVID_8TB/vlp_results/selected_MedViLL_seed777.json', 'w') as f:
    json.dump(json_file, f)