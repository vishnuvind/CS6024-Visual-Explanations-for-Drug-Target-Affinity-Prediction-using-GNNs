import os
import sys
if sys.path[-1] != os.getcwd():
    sys.path.append(os.getcwd())

import time
import json

from log.basic_logger import BasicLogger

def create_dir(dir_list):
    assert  isinstance(dir_list, list) == True
    for d in dir_list:
        if not os.path.exists(d):
            os.makedirs(d)

class ExplainLogger(BasicLogger):
    def __init__(self, args):
        self.args = args
        save_dir = args.get('save_dir')
        if save_dir == None:
            raise Exception('save_dir can not be None!')
        
        self.log_save_dir = os.path.join(save_dir, args.get('saved_model'))
        self.drug_dir = os.path.join(self.log_save_dir, 'drug')
        self.target_dir = os.path.join(self.log_save_dir, 'target')
        create_dir([self.drug_dir, self.target_dir])

        print(self.log_save_dir)
        log_path = os.path.join(self.log_save_dir, 'explain.log')
        super().__init__(log_path)

    def get_drug_dir(self):
        if hasattr(self, 'drug_dir'):
            return self.drug_dir
        else:
            return None

    def get_target_dir(self):
        if hasattr(self, 'target_dir'):
            return self.target_dir
        else:
            return None