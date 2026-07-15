import os
for root, dirs, files in os.walk('C:/team/chym_aki'):
    if 'descriptor_labels.csv' in files:
        print(os.path.join(root, 'descriptor_labels.csv'))
