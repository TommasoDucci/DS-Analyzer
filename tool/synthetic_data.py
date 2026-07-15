import os
import time
import sys
import torch
import threading
import multiprocessing

class SyntheticClassificationIterator(object):    
    def __init__(self, args):  
        self.args = args
        self.iter_num = -1
        self.dev = self.args.local_rank
        self.tensor_type = getattr(args, 'tensor_type', 'img')

        if self.args.precreate:
            if self.tensor_type == 'img':
                self.tensor_bank = load_tensors(self.args.tensor_path, self.args.num_minibatches)
            elif self.tensor_type == 'seq':
                self.tensor_bank = load_mlm_tensor(self.args.tensor_path, self.args.num_minibatches)
            else:
                raise ValueError("Unknown tensor_type: {}".format(self.tensor_type))
        else:
            if self.tensor_type == 'img':
                self.tensor_bank = get_image_classification_tensors(self.args.batch_size, self.args.num_minibatches, self.args.classes)
            elif self.tensor_type == 'seq':
                self.tensor_bank = get_mlm_tensors(self.args.batch_size, self.args.num_minibatches, self.args.tokenizer,self.args.mask_rate)
            else:
                raise ValueError("Unknown tensor_type: {}".format(self.tensor_type))
        print("Got {} tensors".format(len(self.tensor_bank.keys()))) 
        if len(self.tensor_bank.keys()) == 0:
            raise Exception("Could not precreate tensors!")   
    @property
    def _size(self):
        return self.args.num_minibatches*self.args.batch_size

    def __iter__(self):
        return self
 
    def __next__(self):     
        self.iter_num += 1
        if self.tensor_type == 'img':
            images, target = self.iter_image() 
            return images,target
        elif self.tensor_type == 'seq':
            mini_batch = self.iter_mlm()
            return mini_batch
    
    def iter_image(self):
        if self.iter_num < self.args.num_minibatches:   
            self.args.dprof.start_memcpy_tick()  
            images, target = self.tensor_bank[self.iter_num] 
            images = images.cuda(self.dev) 
            target = target.cuda(self.dev) 
            self.args.dprof.stop_memcpy_tick()  
            return images, target   
        else:        
            raise StopIteration 
    
    def iter_mlm(self):
        if self.iter_num < self.args.num_minibatches:    
            mini_batch = self.tensor_bank[self.iter_num]   
            return mini_batch   
        else:        
            raise StopIteration 

    def next(self):      
        return self.__next__()   

def load_tensors(path, total):
    s = time.time()   
    tensor_bank = {}
    th = []
    for i in range(0,5):
        th.append(threading.Thread(target=_load, args=(i*int(total/5), int(total/5), path, tensor_bank)))
        th[i].start()
    
    for i in range(0,5):
        th[i].join()
    print("Loaded {} tensors in {} s".format(len(tensor_bank.keys()), time.time() - s)) 
    return tensor_bank

def _load(start, count, path, tensor_bank):
    for i in range(start, start+count):
        img_name = path + "/image-" + str(i) + ".pt"
        label_name = path + "/label-" + str(i) + ".pt"
        image = torch.load(img_name)
        label = torch.load(label_name)
        tensor_bank[i] = (image, label)


def load_mlm_tensor(path, total):
    s = time.time()
    tensor_bank = {}
    th = []
    for i in range(0,5):
        th.append(threading.Thread(target=_load_sequence, args=(i*int(total/5), int(total/5), path, tensor_bank)))
        th[i].start()

    for i in range(0,5):
        th[i].join()
    print("Loaded {} tensors in {} s".format(len(tensor_bank.keys()), time.time() - s))
    return tensor_bank

def _load_sequence(start, count, path, tensor_bank):
    for i in range(start, start+count):
        seq_name = path + "/seq-" + str(i) + ".pt"
        label_name = path + "/label-" + str(i) + ".pt"
        seq = torch.load(seq_name)
        label = torch.load(label_name)
        tensor_bank[i] = (seq, label)


def get_shared_image_classification_tensors(batch_size, iters, start, num_classes=1000, path="./train"):
    print("Pre-populating train tensors ...")   
    s = time.time()   
    for i in range(start, start+iters):     
        img_name = path + "/image-" + str(i) + ".pt"
        label_name = path + "/label-" + str(i) + ".pt"
        img = getRandImgClassificationTensor(batch_size) 
        target = getRandTargetClassificationTensor(batch_size, num_classes) 
        torch.save(img, img_name)
        torch.save(target, label_name)
    print("Created {} tensors in {} s".format(iters, time.time() - s))


def get_shared_sequence_classification_tensors(batch_size, iters, start, num_classes=1000, path="./train"):
    print("Pre-populating train tensors ...")
    s = time.time()
    for i in range(start, start+iters):
        seq_name = path + "/seq-" + str(i) + ".pt"
        label_name = path + "/label-" + str(i) + ".pt"
        seq = getMLMTensor(batch_size)
        target = getRandTargetClassificationTensor(batch_size, num_classes)
        torch.save(seq, seq_name)
        torch.save(target, label_name)
    print("Created {} tensors in {} s".format(iters, time.time() - s))



def get_image_classification_tensors(batch_size, iters, num_classes=1000):
    print("Pre-populating train tensors ...")   
    tensor_bank={}     
    s = time.time()   
    for i in range(0, iters):     
    #for i in range(start, start+iters):     
        img = getRandImgClassificationTensor(batch_size) 
        target = getRandTargetClassificationTensor(batch_size, num_classes) 
        tensor_bank[i] = (img,target)          
    print("Created {} tensors in {} s".format(iters, time.time() - s))
    return tensor_bank


def get_mlm_tensors(batch_size,num_minibatches,tokenizer,mask_rate = 0.15):
    print("Pre-populating train tensors ...")
    tensor_bank = {}
    s = time.time()
    for i in range(0, num_minibatches):
        mini_batch = getMLMTensor(batch_size, tokenizer, mask_rate)
        tensor_bank[i] = mini_batch
    print("Created {} tensors in {} s".format(num_minibatches, time.time() - s))
    return tensor_bank

def getRandImgClassificationTensor(batchsize):
    return torch.randn(batchsize, 3, 224, 224)

def getMLMTensor(batchsize, tokenizer, mask_rate, seq_len=4096):
    input_ids = torch.randint(0, tokenizer.vocab_size, (batchsize, seq_len), dtype=torch.long)
    attn_mask = torch.ones((batchsize, seq_len))
    masked = torch.rand(batchsize,seq_len) < mask_rate
    mask_id = tokenizer.mask_token_id

    labels = input_ids.clone()    
    labels[~masked] = -100        
    input_ids[masked] = mask_id

    mini_batch = {"input_ids" : input_ids,"attention_mask" : attn_mask,"labels" : labels}

    return mini_batch

def getRandTargetClassificationTensor(batchsize, num_classes):
    return torch.randint(0, num_classes, (batchsize,), dtype=torch.long)

