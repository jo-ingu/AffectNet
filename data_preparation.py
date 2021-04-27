import numpy as np
import os
import csv

def process_labels(file_path):
    # make lists of labels and paths
    labels = []
    paths = []
    with open(file_path, 'r') as csv_file:
        r = csv.reader(csv_file, delimiter=',')
        count = 0
        
        for row in r:
            if 'sub' in row[0]:
                continue
            emotion = int(row[-3]) # emotion -> 0~7
            path = 'C:/Users/VBLAB/Desktop/jo/AffectNet/Manually_Annotated_Images/' + row[0]
            
            if emotion > 7:
                # 8 : None, 9 : Uncertain, 10 : Non-Face
                continue
            
            if not os.path.exists(path):
                print('error: no image')
                continue
            
            labels.append(emotion)
            paths.append(path)
            count += 1
            print('Loaded:', count, end='\r')
    
    print('Loaded:', count)
    return paths, labels
    
if __name__ == '__main__':
    # Save
    train_paths, train_labels = process_labels('Desktop/jo/AffectNet/training.csv')
    np.save('Desktop/jo/AffectNet/training_paths', train_paths)
    np.save('Desktop/jo/AffectNet/training_labels', train_labels)
    
    validation_paths, validation_labels = process_labels('Desktop/jo/AffectNet/validation.csv')
    np.save('Desktop/jo/AffectNet/validation_paths', validation_paths)
    np.save('Desktop/jo/AffectNet/validation_labels', validation_labels)
