import rocksdb 
import numpy as np
# GPT4 recommended solution for pickle5 / pickle imports when used with different containers
import sys
if sys.version_info < (3, 8):
    import pickle5 as mypickler
else:
    import pickle as mypickler

sys.path.append('/home/potzschf/repos/')
from helperToolz.feevos.xlogger import *  # Necessary for saving keys 
import os 
import ast # Reads keys
import pandas as pd # Reads keys

# Parallel WriteBatch
from pathos.pools import ThreadPool as pp 
from multiprocessing import Lock

from collections import OrderedDict
import rasterio
from math import ceil as mceil

class _RocksDBBase(object):
    """
    Base class with useful defaults 
    Creates a database with two families (columns), of inputs and labels 
    """
    def __init__(self, 
                 lru_cache=1,
                 lru_cache_compr=0.5,
                 num_workers = 16,
                 read_only=True):
        super().__init__()

        GB = 1024**3  # 1 GB in bytes
        self.lru_cache_GB       = lru_cache         * GB  # Convert lru_cache from GB to bytes
        self.lru_cache_compr_GB = lru_cache_compr   * GB  # Convert lru_cache_compr from GB to bytes
        self.num_workers        = num_workers
        self.read_only          = read_only
        
    def _get_db_opts_default(self):
        opts = rocksdb.Options()
        opts.create_if_missing = True
        opts.max_open_files = 3e+5        # Some Defaults 
        opts.write_buffer_size = 67108864 # Some Defaults 
        opts.max_write_buffer_number = 30 # 3 default
        opts.target_file_size_base = 67108864  # default 67108864, value starting 7: 7340032 input.nbytes
        #opts.paranoid_checks=False

        # @@@@@@@@@@@ performance boost @@@@@@@@@@
        opts.IncreaseParallelism(self.num_workers)
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


        opts.table_factory = rocksdb.BlockBasedTableFactory(
                filter_policy=rocksdb.BloomFilterPolicy(10), # 10 default for 1% error 
                block_cache=rocksdb.LRUCache(self.lru_cache_GB), # 16GB
                block_cache_compressed=rocksdb.LRUCache(self.lru_cache_compr_GB )) # 0.5 GB 

        return opts

    def _get_db_colfamopts_default(self):
        opts = rocksdb.ColumnFamilyOptions()
        opts.write_buffer_size = 67108864 # Some Defaults 
        opts.max_write_buffer_number = 30 # 3 default
        opts.target_file_size_base = 67108864  # default 67108864, value starting 7: 7340032 input.nbytes
        #opts.paranoid_checks=False


        opts.table_factory = rocksdb.BlockBasedTableFactory(
                filter_policy=rocksdb.BloomFilterPolicy(10), # 10 default for 1% error 
                block_cache=rocksdb.LRUCache( self.lru_cache_GB ), # 16GB
                block_cache_compressed=rocksdb.LRUCache( self.lru_cache_compr_GB)) # 0.5 GB 

        return opts
        
    def _open_rocks_write(self,flname_db):
        # Remark: it is fastest if I bundle together inputs and labels, though not generic

        self.opts_db = self._get_db_opts_default()
        self.db = rocksdb.DB(flname_db, self.opts_db, read_only=self.read_only)


        self.cf_opts    = OrderedDict()
        self.cf_objects = OrderedDict()
        for family_name in self.cf_names:
            self.cf_opts[family_name]  = self._get_db_colfamopts_default()
            self.cf_objects[family_name] = self.db.create_column_family(family_name,self.cf_opts[family_name])

    def _open_rocks_read(self,flname_db):
        # Remark: it is fastest if I bundle together inputs and labels, though not generic

        self.opts_db = self._get_db_opts_default()

        self.cf_opts    = OrderedDict()
        for family_name in self.cf_names:
            self.cf_opts[family_name]  = self._get_db_colfamopts_default() # 


        self.db = rocksdb.DB(flname_db, self.opts_db, column_families={b'default':self.opts_db, **self.cf_opts},read_only=self.read_only)

class RocksDBWriter(_RocksDBBase):
    """
    flname_db: Filename for where the database will be written. 
    metadata: anything you want to save to the database (under key: 'meta') along with the data 

    It should contain at a minimum the following keys: inputs_shape, inputs_dtype, labels_shape, labels_dtype. E.g. 
    
    metadata={'inputs':{
                'inputs_shape':(14,256,256),
                'inputs_dtype':np.float32},

                'labels':{'labels_shape':(NClasses,256,256), # Use None for Integer class labels 
                'labels_dtype':np.uint8}
                }
    """
    def __init__(self, 
                 flname_db, 
                 metadata, # Dict of dicts (ordered). 
                 lru_cache=1,
                 lru_cache_compr=0.1,
                 num_workers = 8,
                 read_only=False):
        super().__init__(lru_cache,lru_cache_compr,num_workers,read_only)
        
        # Define Column families 
        self.cf_names = [str.encode(key) for key in metadata.keys()]
    
        #Open Database
        self._open_rocks_write(flname_db)


        # Write meta dictionary in file
        with open(os.path.join(flname_db,'metadata.dat'), 'wb') as handle:
            mypickler.dump(metadata, handle, protocol=mypickler.HIGHEST_PROTOCOL) # Works only for mypickler == pickle  


        # Write in database as well, legacy operation 
        meta_key = b'meta'
        meta_dumps = mypickler.dumps(metadata)
        self.db.put(meta_key,meta_dumps)
              
        # Initialize ascii file that has all keys 
        # Writes all keys EXCEPT b'meta' 
        self.keys_logger = xlogger(os.path.join(flname_db,'keys.dat'))

        self.global_idx = 0
        self.lock = Lock()

    def write_batch(self,batch):
        # batch: iterable of tuples of numpy arrays 
        wb = rocksdb.WriteBatch()

        # Parallel writing of batch of data 
        def writebatch(global_idx,datum):
            key_input = '{}'.format(global_idx).encode('ascii')

            for cfname, tinput in zip(self.cf_names, datum):
                cfinputs = self.db.get_column_family(cfname)
                wb.put((cfinputs ,key_input),  tinput.tobytes())
               
            # Write keys into file, for fast accessing them 
            self.lock.acquire()
            self.keys_logger.write({'keys':key_input})
            self.lock.release()


        pool = pp(nodes=self.num_workers) # with 16 works nice
        global_indices = [self.global_idx + i for i in range(len(batch))]
        
        
        result = pool.map(writebatch,global_indices,batch)
        # Write batch 2 DB
        self.db.write(wb)
        # Update lglobal index 
        self.global_idx = global_indices[-1]+1

class RocksDBReader(_RocksDBBase):
    def __init__(self, 
                 flname_db, 
                 lru_cache=1,
                 lru_cache_compr=0.5,
                 num_workers = 4,
                 read_only=True):
        super().__init__(lru_cache,lru_cache_compr,num_workers,read_only)



        # Read meta dictionary in file
        path = os.path.join(flname_db,'metadata.dat')
        if os.path.exists(path):
            with open( path, 'rb') as handle:
                self.meta = mypickler.load(handle) # Works only for mypickler == pickle  
            # Define Column families 
            self.cf_names = [str.encode(key) for key in self.meta.keys()]
            self._open_rocks_read(flname_db)
        else: # Legacy 
            #Open Database
            self._open_rocks_read(flname_db)
            meta = self.db.get(b'meta')
            self.meta = mypickler.loads(meta) 
            self.cf_names = {b'inputs',b'labels'}

        # Read all keys 
        self.keys = self._read_keys(flname_db)


    def _read_keys(self,flname_db, flname_keys = 'keys.dat',sep="|",lineterminator='\n'):
        # This function works in conjuction with the RocksDBDatasetWriter class, and reads defaults 
        flname_keys = os.path.join( flname_db, flname_keys) 
        df = pd.read_csv(flname_keys ,sep=sep,lineterminator=lineterminator)
        df['keys'] = df['keys'].apply(lambda x: ast.literal_eval(x))
        return df['keys'].tolist()


    def get_inputs_labels(self,idx):

        key = self.keys[idx]

        all_inputs = []
        for cname in self.cf_names:
            tcfinputs = self.db.get_column_family(cname)
            cname = bytes.decode(cname)
            tinputs = self.db.get( (tcfinputs, key) )
            tshape =  self.meta[cname]['{}_shape'.format(cname)]
            if tshape is not None:
                # print(np.frombuffer(tinputs, dtype= self.meta[cname]['{}_dtype'.format(cname)]).shape)
                tinputs = np.frombuffer(tinputs, dtype= self.meta[cname]['{}_dtype'.format(cname)]).reshape(*tshape)

            else:
                tinputs = np.frombuffer(tinputs, dtype= self.meta[cname]['{}_dtype'.format(cname)])
              
            all_inputs.append(tinputs.copy()) # The np.frombuffer results in a readonly array, that forces pytorch to create warnings. This is usually taken care of in transform method during training
            #all_inputs.append(tinputs) # 


        return all_inputs


### Convenience class that writes data into database
from time import time
from datetime import timedelta
from helperToolz.feevos.rasteriter import RasterMaskIterableInMemory 
from helperToolz.feevos.progressbar import progressbar
class Rasters2RocksDB(object):
    def __init__(self, 
                 lstOfTuplesNames, 
                 names2raster_function, 
                 metadata, 
                 flname_prefix_save, 
                 transformT=None,
                 transformV=None, 
                 # Some useful defaults for Remote Sensing (large) imagery
                 batch_size=2,  
                 Filter=256,
                 stride_divisor=2,
                 train_split=0.9,
                 split_type='sequential'):
        super().__init__()
        
        self.listOfTuplesNames = lstOfTuplesNames
        self.names2raster = names2raster_function
        flname_db_train = os.path.join(flname_prefix_save,'train.db') 
        flname_db_valid = os.path.join(flname_prefix_save,'valid.db')

        self.Filter = Filter
        self.stride_divisor = stride_divisor

        self.dbwriter_train = RocksDBWriter(flname_db_train,metadata)
        self.dbwriter_valid = RocksDBWriter(flname_db_valid,metadata)

        
        self.transformT   = transformT
        self.transformV   = transformV
        self.batch_size  = batch_size
        self.train_split = train_split
        split_types = ['sequential','random']
        assert split_type in split_types, ValueError("Cannot understand split_type, available options::{}, aborting ...".format(split_types))
        self.split_type = split_type

        print ("Creating databases in location:{}".format(flname_prefix_save))
        print('Database train::{}'.format(flname_db_train))
        print('Database valid::{}'.format(flname_db_valid))

    def write_split_strategy(self,batch_idx, NTrain_Total, some_batch):
        if self.split_type == 'random':
            if np.random.rand() < self.train_split:
                self.dbwriter_train.write_batch(some_batch)
            else:
                self.dbwriter_valid.write_batch(some_batch)
        elif self.split_type == 'sequential':
                if batch_idx < NTrain_Total:
                    self.dbwriter_train.write_batch(some_batch)
                else:
                    self.dbwriter_valid.write_batch(some_batch)

    def create_dataset(self):
        # For all triples in list of filenames                      
        tic = time()                                                
        for idx,names in enumerate(self.listOfTuplesNames):           
            print ("============================")                  
            print ("----------------------------")                  
            print ("Processing:: {}/{} Tuple".format(idx+1, len(self.listOfTuplesNames)))
            print ("----------------------------")                  
            for name in names:                                      
                print("Processing File:{}".format(name))            
            print ("****************************")

            lst_of_rasters = self.names2raster(names)
            myiterset = RasterMaskIterableInMemoryFP(lst_of_rasters,
                                                   Filter = self.Filter,
                                                   stride_divisor=self.stride_divisor ,
                                                   transformT=self.transformT,
                                                   transformV=self.transformV,
                                                   batch_size=self.batch_size,
                                                   batch_dimension=False)

            nbset = myiterset.get_len_batch_set()
            train_split = int(self.train_split*nbset)
            for idx2 in progressbar(range(nbset)):
                batch = myiterset.get_batch(idx2, train_split)
                self.write_split_strategy(idx2,train_split,batch)

        Dt = time() - tic                                           
        Dt = str(timedelta(seconds=Dt))
        NData = len(self.listOfTuplesNames)
        print("time to WRITE N::{} files, Dt::{}".format(NData,Dt)) 
        
            
        print (" XXXXXXXXXXXXXXXXXXXXXXX Done! XXXXXXXXXXXXXXXXXXXXXX")
                

#### TORCH Dataset Class 
import torch
class RocksDBDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 flname_db,
                 transform=None,
                 lru_cache=0.01,
                 lru_cache_compr=0.01,
                 num_workers = 5,
                 read_only=True):
        super().__init__()

        self.mydbreader = RocksDBReader(flname_db,lru_cache,lru_cache_compr,num_workers,read_only)
        self.length = len(self.mydbreader.keys)
        self.transform = transform
    
    def __len__(self):
        return self.length

    def __getitem__(self,idx):

        data = self.mydbreader.get_inputs_labels(idx)
        
        if self.transform is not None:
            data = self.transform(*data)
            return data

        return data

class _RasterIterableBase(object):
    # An iterable version of raster sliding window patches 
    # Shape assumes Channels x Height x Width format 

    def __init__(self, shape, Filter=256, stride_divisor=2, batch_size=None):
        super().__init__()

        self.shape = shape
        self.F = Filter
        self.s = Filter//stride_divisor
        self.generate_slices()
        if batch_size is not None:
            self.batch_size = batch_size
            self.batchify(batch_size)

    def get_len_set(self):
        return len(self.RowsCols)

    def get_len_batch_set(self):
        return len(self.BatchRowsCols)

    def generate_slices(self):

        shape = self.shape 
        # Constants that relate to rows, columns 
        self.nTimesRows = int((shape[-2] - self.F)//self.s + 1)
        self.nTimesCols = int((shape[-1] - self.F)//self.s + 1)

        # Use these directly 
        RowsCols = [(row, col) for row in range(self.nTimesRows-1) for col in range(self.nTimesCols-1)]
        RowsCols_Slices = [ (slice(row*self.s,row*self.s +self.F,1),slice(col*self.s,col*self.s+self.F,1) )  for (row,col) in RowsCols ]

        # Construct RowsCols for last Col 
        col_rev = shape[-1]-self.F
        Rows4LastCol = [(row,col_rev) for row in range(self.nTimesRows-1)]
        Rows4LastCol_Slices = [ (slice(row*self.s,row*self.s +self.F,1),slice(col_rev,col_rev+self.F,1) )  for (row,col_rev) in Rows4LastCol]

        # Construct RowsCols for last Row 
        row_rev = shape[-2]-self.F
        Cols4LastRow        = [(row_rev,col) for col in range(self.nTimesCols-1)]
        Cols4LastRow_Slices = [(slice(row_rev,row_rev+self.F,1),slice(col*self.s,col*self.s +self.F,1) )  for (row_rev,col) in Cols4LastRow]

        
        # Store all Rows and Columns that correspond to raster slices and slices 
        self.RowsCols           = RowsCols + Rows4LastCol + Cols4LastRow
        self.RowsCols_Slices    = RowsCols_Slices + Rows4LastCol_Slices + Cols4LastRow_Slices


    def batchify(self,batch_size):
        n = mceil(len(self.RowsCols)/batch_size)
        self.BatchIndices  = np.array_split(list(range(len(self.RowsCols))),n,axis=0)
        self.BatchRowsCols = np.array_split(self.RowsCols,n,axis=0)
        self.BatchRowsCols_Slices = np.array_split(self.RowsCols_Slices,n,axis=0)

class RasterMaskIterableInMemoryFP(_RasterIterableBase):
    # This will accept read in windows from Rasterio 

    def __init__(self, lst_of_rasters, Filter=256, stride_divisor=2, transformT=None, transformV=None,  batch_size=None, num_workers=28, 
            batch_dimension=False):
        self.lst_of_rasters = lst_of_rasters
        assert len(lst_of_rasters) >= 2, ValueError("You need at least two files, an input image and a target mask, you provided::{}".format(self.number_of_rasters))
        shape = lst_of_rasters[0].shape

        for idx in range(1,len(lst_of_rasters)):
            assert shape[-2:] == lst_of_rasters[idx].shape[-2:], ValueError("All rasters in the list must have the same spatial dimensionality ")

        super().__init__(shape=shape, Filter=Filter, stride_divisor=stride_divisor, batch_size=batch_size)
        self.transformT   = transformT
        self.transformV   = transformV
        self.num_workers = num_workers

        self.batch_dimension = batch_dimension

    def get_batch(self, idx, NTrain_Total):
        batch_patches = []
        for slice_row,slice_col in self.BatchRowsCols_Slices[idx]: # Batch Indices 
            patches = []
            for raster in self.lst_of_rasters:
                patches.append(raster[...,slice_row,slice_col][None]) # Add batch dimension
                #patches.append(raster[...,slice_row,slice_col])
            batch_patches.append(patches)

        if self.transformT is not None:
            # Trick to go from list to arguments
            if idx < NTrain_Total:
                def vtransformT(patch):
                    tpatch = [p[0] for p in patch] # Remove batch dim for transform
                    # if len(np.unique(tpatch[1])) == 1:
                    #     next
                    # else:
                    tpatch2 = self.transformT(*tpatch)
                    if len(np.unique(tpatch2[1])) == 1:
                        return tpatch
                    else:
                        #tpatch = [p[None] for p in tpatch] # Restore batch dim
                        return tpatch2
                pool = pp(nodes=self.num_workers)
                result = pool.map(vtransformT,batch_patches)
                batch_patches = result
            else: # only do normalisation stuff
                def vtransformV(patch):
                    tpatch = [p[0] for p in patch] # Remove batch dim for transform
                    # if len(np.unique(tpatch[1])) == 1:
                    #     next
                    # else:
                    tpatch2 = self.transformV(*tpatch)
                    #tpatch = [p[None] for p in tpatch] # Restore batch dim
                    if len(np.unique(tpatch2[1])) == 1:
                        return tpatch
                    else:
                        return tpatch2
                pool = pp(nodes=self.num_workers)
                result = pool.map(vtransformV,batch_patches)
                batch_patches = result

        if not self.batch_dimension:
            return batch_patches
        # Now concatenate all along the first dimension?
        lst_of_elements_in_patch = zip(*batch_patches)
        batched_elements = []
        for tinput in lst_of_elements_in_patch:
            tinput = [t[None] for t in tinput] # Add batch dimension
            batched_elements.append(np.concatenate(tinput,axis=0))


        return batched_elements
