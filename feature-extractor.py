import os
import sys
import numpy as np
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

sys.path.insert(0, '/home/ubuntu/caffe/python')
import caffe

caffeRootDir = '/home/ubuntu/caffe/'

# Load pre-trained model
model_file = caffeRootDir + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
deploy_prototxt = caffeRootDir + 'models/bvlc_reference_caffenet/deploy.prototxt'
net = caffe.Net(deploy_prototxt, model_file, caffe.TEST)
layer = 'fc7'
if layer not in net.blobs:
    raise TypeError("Invalid layer name: " + layer)

# Define the transformer
imagemean_file = caffeRootDir + 'models/bvlc_reference_caffenet/ilsvrc_2012_mean.npy'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load(imagemean_file).mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))
transformer.set_raw_scale('data', 255.0)

# Load and normalize dataset
directory = caffeRootDir + 'examples/authorship/data/wiertz/'
data_pos = np.array([])
for filename in os.listdir(directory):
    if filename.endswith(".jpg"): 
        filepath = os.path.join(directory, filename)
        print(filepath)
        net.blobs['data'].reshape(1,3,227,227)
        img = caffe.io.load_image(filepath)
        net.blobs['data'].data[...] = transformer.preprocess('data', img)
        
        # Network forward (feature calculation)
        output = net.forward()
        feature_vector = net.blobs[layer].data[0];
        print(feature_vector)
#        data_pos = np.hstack((data_pos, np.transpose(feature_vector)))
        data_pos = np.append(data_pos, feature_vector)
#        data_pos = np.append(data_pos, 1)
#        data_pos = np.concatenate((data_pos, np.transpose(feature_vector)))
        continue
    else:
        continue

directory = caffeRootDir + 'examples/authorship/data/rodin/'
data_neg = np.array([])
for filename in os.listdir(directory):
    if filename.endswith(".jpg"): 
        filepath = os.path.join(directory, filename)
        print(filepath)
        net.blobs['data'].reshape(1,3,227,227)
        img = caffe.io.load_image(filepath)
        net.blobs['data'].data[...] = transformer.preprocess('data', img)
        
        # Network forward (feature calculation)
        output = net.forward()
        feature_vector = net.blobs[layer].data[0];
        print(feature_vector)
#        data_neg = np.hstack((data_neg, np.transpose(feature_vector)))
        data_neg = np.append(data_neg, feature_vector)
#        data_neg = np.append(data_neg, 1)
#        data_neg = np.concatenate((data_neg, np.transpose(feature_vector)))
        continue
    else:
        continue

# Train linear classifier and test
print(np.shape(data_pos))
print(np.shape(data_neg))
data_pos = np.reshape(data_pos, (20, 4096))
data_neg = np.reshape(data_neg, (35, 4096))
print(np.shape(data_pos))
print(np.shape(data_neg))

X_train = np.vstack([data_pos[0:14, :], data_neg[0:24, :]])
y_train = np.vstack([np.zeros(shape=(14,1)), np.ones(shape=(24,1))])
print(np.shape(X_train))
print(np.shape(y_train))

X_test = np.vstack([data_pos[15:, :], data_neg[25:, :]])
y_test = np.vstack([np.zeros(shape=(5,1)), np.ones(shape=(10,1))])
print(np.shape(X_test))
print(np.shape(y_test))

print "================== RandomForestClassifier =================="
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print classification_report(y_test, y_pred, target_names=['Antoine Wiertz', 'Auguste Rodin'])
print confusion_matrix(y_test, y_pred)

print "================== svm =================="
#svm = svm.SVC(kernel='linear', C=1)
svm = svm.SVC(kernel='rbf', C=1)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print classification_report(y_test, y_pred, target_names=['Antoine Wiertz', 'Auguste Rodin'])
print confusion_matrix(y_test, y_pred)

print "================== DecisionTreeClassifier =================="
dt = tree.DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print classification_report(y_test, y_pred, target_names=['Antoine Wiertz', 'Auguste Rodin'])
print confusion_matrix(y_test, y_pred)
