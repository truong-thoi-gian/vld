#!/usr/bin/env python

import tflearn
from numpy import argmax
from sklearn import metrics
from sklearn.utils import shuffle
import json
import numpy as np

import train
params = train.params

import json
import shutil
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import re
import subprocess
from functools import reduce

import tensorflow as tf

import itertools
from sklearn.utils import shuffle
from sklearn import model_selection, metrics
from datetime import datetime
# import predictbyyara
# import keras
import copy
model_config = 'config.json'

def serialize_decorator(func):
    global params

    with open(params['opcodes'], 'r') as f:
        code_record = list(map(lambda x: x.strip(), f.readlines()))

    def wrapped(*args, **kwargs):
        return func(*args, _code_record=code_record, **kwargs)
    return wrapped


@serialize_decorator
def serialize_codes(code_list, _code_record):
    for file_code in code_list:
        for index, code in enumerate(file_code):
            if _code_record.count(code):
                file_code[index] = _code_record.index(code) + 1
            else:
                file_code[index] = 0

def get_php_file(base_dir):
    file_list = []
    for path, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.php'):
                filename = os.path.realpath(os.path.join(path, file))
                file_list.append(filename)
    return file_list


def get_all_opcode(base_dir):
    file_list = get_php_file(base_dir)
    opcode_list = []
    all_file = []
    iter = 0
    if (len(file_list) > 1000):
        print("file list is too long ", len(file_list))
    for file in file_list[0:1000]:
        opcode = get_file_opcode(file)
        #print("file opcode",file,opcode)
        if (opcode and opcode not in opcode_list):
            opcode_list.append(opcode)
	    all_file.append(file)
            #shutil.copyfile(file, base_dir + '/' + str(iter) + '.php')
            # print(file + 'move to' + base_dir + '/' + str(iter) + '.php')
            iter = iter + 1
    return [all_file,opcode_list]


def get_file_opcode(filename):
    #cmd = ('docker run -v {0}:/src/index.php php_vld'.format(filename)).split() # '/usr/bin/php'
    cmd = ('{0} -dvld.active=1 -dvld.execute=0'.format("/usr/local/bin/php")).split() #('{0} -dvld.active=1 -dvld.execute=0'.format(php_path)) #
    cmd.append(filename)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    opcodes = []
    pattern = r'([A-Z_]{2,})\s+'
    #print "Getting opcodes of %s" %(cmd)
#     print(p.stdout.readlines())
#     output = p.stdout.readlines().strip('\r\n')
#                     json_output = json.loads(output)
#                     if 'stream' in json_output:
#                         click.echo(json_output['stream'].strip('\n'))
    for line in p.stdout.readlines()[8:-3]:
        #print(line)
        try:
            match = re.search(pattern, line.decode("utf-8"))
        except UnicodeDecodeError:
            match = re.search(pattern, str(line))
        if match:
            opcodes.append(match.group(1))
    p.terminate()
    return opcodes

if __name__ == '__main__':

    #white_code_list = json.load(open(params['test_white_list'], 'r'))
    #test_2 = [[72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 188, 72, 72, 72, 72]]

    all_file,x_test = get_all_opcode('/codes')
    #print("list is",x_test, all_file)
    original_list = copy.deepcopy(x_test)# np.array(x_test, copy=True) #np.copy(x_test)
    serialize_codes(x_test)
    #print("encoded codes is ",x_test, "vs ", [white_code_list[2]])
    #print('Testing set size: {0}'.format(len(x_test)))

    # print('Serializing opcodes...')
    # train.serialize_codes(x_test)   
    x_test = tflearn.data_utils.pad_sequences(x_test, maxlen=train.params['seq_length'], value=0)
    #white_code_list = tflearn.data_utils.pad_sequences(white_code_list, maxlen=train.params['seq_length'], value=0)
    #test_2 = tflearn.data_utils.pad_sequences(test_2, maxlen=train.params['seq_length'], value=0)
    
    #print('Creating network...')
    network = train.create_network(
        train.params['seq_length'], 
        learning_rate=train.params['learning_rate'])
    #print('Done! Doing DNN...')

    model = tflearn.DNN(network, tensorboard_verbose=0)
    #print("Done! Loading model %s..." %(train.params['model']))
    model.load(train.params['model'])
    
    print('Done! Predicting ...')

    sample = x_test #[x_test[1],test_2[0],white_code_list[1]]#x_test
    p = model.predict(sample)
    q = np.argmax(p, axis=1)
    max_index = np.argmax(q)
    result = max(q)
    if (result):
      print("There is one  webshell in file ", all_file[max_index], " with opcode is ", original_list[max_index], " encoded code is ", x_test[max_index][0:100])
    else:
      print("There is no webshell ")
