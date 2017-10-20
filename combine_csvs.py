import csv

with open('data/mjc+mrg/combined_hal.csv', 'w') as wf:
    with open('data/mjc/driving_log_clean_hal.csv', 'r') as rf1:
        with open('data/mrg/driving_log_clean_hal.csv', 'r') as rf2:
            reader1 = csv.reader(rf1)
            reader2 = csv.reader(rf2)
            writer = csv.writer(wf)
            for row in reader1: writer.writerow(row)
            for row in reader2: writer.writerow(row)
