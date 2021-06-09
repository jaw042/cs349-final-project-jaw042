import numpy as np
import os

base = '/Users/jacobwit/Documents/GitHub/CS-349/final-project'
data_path = os.path.join(base, 'YearPredictionMSD.txt')
all_data = np.loadtxt(data_path, delimiter=',')
print(np.shape(all_data))

# train_base = '/Users/jacobwit/Documents/GitHub/CS-349/final-project/UCI_HAR_Dataset/train'
# test_base = '/Users/jacobwit/Documents/GitHub/CS-349/final-project/UCI_HAR_Dataset/test'
# train_x_path = os.path.join(train_base, 'X_train.txt')
# train_y_path = os.path.join(train_base, 'y_train.txt')
# train_subject_path = os.path.join(train_base, 'subject_train.txt')
# test_x_path = os.path.join(test_base, 'X_test.txt')
# test_y_path = os.path.join(test_base, 'y_test.txt')
# test_subject_path = os.path.join(test_base, 'subject_test.txt')
# train_x = np.loadtxt(train_x_path)
# train_y = np.loadtxt(train_y_path)
# train_subject = np.loadtxt(train_subject_path)
# test_x = np.loadtxt(test_x_path)
# test_y = np.loadtxt(test_y_path)
# test_subject = np.loadtxt(test_subject_path)
#
# print(np.shape(train_x))
# print(np.shape(train_y))
# print(np.shape(train_subject))
#
# values, counts = np.unique(train_subject, return_counts=True)
#
# print(values)
# print(counts)