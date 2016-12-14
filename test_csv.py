import csv
from os import listdir
from os.path import join

base_dir = '/home/matt/DL/Autonomous_Driving/Udacity/CarND-Behavioral-Cloning/data/IMG'

img_files = listdir('data/IMG/')
with open('data/driving_log_test.csv', 'w') as wf:
    with open('data/driving_log_clean.csv', 'r') as rf:
        reader = csv.reader(rf)
        writer = csv.writer(wf)
        for i, row in enumerate(reader):
            if i > 10: break

            rel_path = row[0].split('/')[-1]
            real_path = join(base_dir, rel_path)
            writer.writerow([real_path, row[1]])
