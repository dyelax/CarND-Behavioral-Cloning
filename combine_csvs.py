import csv

with open('data/sdc-lab/mjc+mrg/combined.csv', 'w') as wf:
    with open('data/sdc-lab/mjc/driving_log_clean.csv', 'r') as rf1:
        with open('data/sdc-lab/mrg/driving_log_clean.csv', 'r') as rf2:
            reader1 = csv.reader(rf1)
            reader2 = csv.reader(rf2)
            writer = csv.writer(wf)
            for row in reader1: writer.writerow(row)
            for row in reader2: writer.writerow(row)
