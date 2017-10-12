import csv
from os import listdir
from os.path import join

base_dir = '/Users/matt/Programming/Deep-Learning/Autonomous_Driving/udacity/term1/CarND-Behavioral-Cloning/data/sdc-lab/mrg/IMG'

img_files = listdir('data/sdc-lab/mrg/IMG/')
with open('data/sdc-lab/mrg/driving_log_clean.csv', 'w') as wf:
    with open('data/sdc-lab/mrg/driving_log.csv', 'r') as rf:
        reader = csv.reader(rf)
        writer = csv.writer(wf)
        for row in reader:
            rel_path = row[0].split('\\')[-1]
            if rel_path in img_files:
                real_path = join(base_dir, rel_path)

                writer.writerow([real_path, row[3]])
