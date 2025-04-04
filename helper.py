import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import rasterio, glob, xarray as xr
import os,sys
# import albumentations as A
# from albumentations.core.transforms_interface import  ImageOnlyTransform
# import torch
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader   
# import matplotlib.pyplot as plt
# sys.path.append('../')                                                      
# from tfcl.models.ptavit3d.ptavit3d_dn import ptavit3d_dn       
# from tfcl.nn.loss.ftnmt_loss import ftnmt_loss               
# from tfcl.utils.classification_metric import Classification  
from datetime import datetime
import time
import higra as hg
from osgeo import gdal


def getFilelist(originpath, ftyp, deep = False, order = True):
    out   = []
    if deep == False:
        files = os.listdir(originpath)
        for i in files:
            if i.split('.')[-1] in ftyp:
                if originpath.endswith('/'):
                    out.append(originpath + i)
                else:
                    out.append(originpath + '/' + i)
            # else:
            #     print("non-matching file - {} - found".format(i.split('.')[-1]))
    else:
        for path, subdirs, files in os.walk(originpath):
            for i in files:
                if i.split('.')[-1] in ftyp:
                    out.append(os.path.join(path, i))
    if order == True:
        out = sorted(out)
    return out

def export_np_to_tif(arr, src, path, name):
    with rasterio.open(
            path + name + '.tif',
            'w',
            crs=None,#src.crs,
            nodata=None, # change if data has nodata value
            transform=src.transform,
            driver='GTiff',
            height=arr.shape[1],
            width=arr.shape[2],
            count=arr.shape[0],
            dtype=arr.dtype
        ) as dst:
            for i in range(arr.shape[0]):
                dst.write(arr[i], i + 1)

def export_single_np_to_tif(arr, src, path, name):
    with rasterio.open(
            path + name + '.tif',
            'w',
            crs=src.crs if src else None,
            nodata=None,
            transform=src.transform if src else None,
            driver='GTiff',
            height=arr.shape[0],
            width=arr.shape[1],
            count=1,
            dtype=arr.dtype
        ) as dst:
        dst.write(arr, 1)     
# Normalization and transform functions

# class AI4BNormal_S2(object):
#     """
#     class for Normalization of images, per channel, in format CHW 
#     """
#     def __init__(self):

#         self._mean_s2 = np.array([5.4418573e+02, 7.6761194e+02, 7.1712860e+02, 2.8561428e+03 ]).astype(np.float32) 
#         self._std_s2  = np.array( [3.7141626e+02, 3.8981952e+02, 4.7989127e+02 ,9.5173022e+02]).astype(np.float32) 

#     def __call__(self,img):
#         temp = img.astype(np.float32)
#         temp2 = temp.T
#         temp2 -= self._mean_s2
#         temp2 /= self._std_s2

#         temp = temp2.T
#         return temp
    
# class TrainingTransformS2(object):
#     # Built on Albumentations, this provides geometric transformation only  
#     def __init__(self,  prob = 1., mode='train', norm = AI4BNormal_S2() ):
#         self.geom_trans = A.Compose([
#                     A.RandomCrop(width=128, height=128, p=1.0),  # Always apply random crop
#                     A.OneOf([
#                         A.HorizontalFlip(p=1),
#                         A.VerticalFlip(p=1),
#                         A.ElasticTransform(p=1), # VERY GOOD - gives perspective projection, really nice and useful - VERY SLOW   
#                         A.GridDistortion(distort_limit=0.4,p=1.),
#                         A.ShiftScaleRotate(shift_limit=0.25, scale_limit=(0.75,1.25), rotate_limit=180, p=1.0), # Most important Augmentation   
#                         ],p=1.)
#                     ],
#             additional_targets={'imageS1': 'image','mask':'mask'},
#             p = prob)
#         if mode=='train':
#             self.mytransform = self.transform_train
#         elif mode =='valid':
#             self.mytransform = self.transform_valid
#         else:
#             raise ValueError('transform mode can only be train or valid')
            
            
#         self.norm = norm
        
#     def transform_valid(self, data):
#         timgS2, tmask = data
#         if self.norm is not None:
#             timgS2 = self.norm(timgS2)
        
#         tmask= tmask 
#         return timgS2,  tmask.astype(np.float32)

#     def transform_train(self, data):
#         timgS2, tmask = data
        
#         if self.norm is not None:
#             timgS2 = self.norm(timgS2)

#         tmask= tmask 
#         tmask = tmask.astype(np.float32)
#         # Special treatment of time series
#         c2,t,h,w = timgS2.shape
#         #print (c2,t,h,w)              
#         timgS2 = timgS2.reshape(c2*t,h,w)
#         result = self.geom_trans(image=timgS2.transpose([1,2,0]),
#                                  mask=tmask.transpose([1,2,0]))
#         timgS2_t = result['image']
#         tmask_t  = result['mask']
#         timgS2_t = timgS2_t.transpose([2,0,1])
#         tmask_t = tmask_t.transpose([2,0,1])
        
#         c2t,h2,w2 = timgS2_t.shape

        
#         timgS2_t = timgS2_t.reshape(c2,t,h2,w2)
#         return timgS2_t,  tmask_t
#     def __call__(self, *data):
#         return self.mytransform(data)

# class VALIDataset(torch.utils.data.Dataset):
#     def __init__(self, path_to_data=r'/../../../../data/fields/', norm=AI4BNormal_S2()):
        
#         # self.flnames_s2_img = sorted(glob.glob(os.path.join(path_to_data,r'images/' + 'LU' + '/*.nc')))
#         # self.flnames_s2_mask = sorted(glob.glob(os.path.join(path_to_data,r'masks/' + 'LU' + '/*.tif')))
    
        
#         self.flnames_s2_img = path_to_data + 'vector/Force_X_from_68_to_69_Y_from_42_to_42.vrt'
#         self.flnames_s2_mask = path_to_data + 'IACS/4_Multitask_labels/GSA-DE_BRB-2019_mtsk.tif'

#         assert len(self.flnames_s2_img) == len(self.flnames_s2_mask), ValueError("Some problem, the masks and images are not in the same numbers, aborting")
        
#         tlen = len(self.flnames_s2_img)
#         self.norm=norm   

#     # Helper function to read nc to raster 
#     def ds2rstr(self,tname):
#         variables2use=['B2','B3','B4','B8'] # ,'NDVI']
#         ds = xr.open_dataset(tname)
#         ds_np = np.concatenate([ds[var].values[None] for var in variables2use],0)

#         return ds_np

#     def read_mask(self,tname):
#         return rasterio.open(tname).read((1,2,3))

#     def __getitem__(self,idx):
#         tname_img = self.flnames_s2_img[idx]
#         tname_mask = self.flnames_s2_mask[idx]
        
#         timg = self.ds2rstr(tname_img)
#         tmask = self.read_mask(tname_mask)
        
#         if self.norm is not None:
#             timg = self.norm(timg)
            
#         return timg, tmask
    
#     def __len__(self):
#         return len(self.flnames_s2_img)

# def plotter(array):

#     # Plot the slices
#     fig, axes = plt.subplots(1, 4, figsize=(20, 6), constrained_layout=False)  # 4 slices
#     slice_indices = np.linspace(0, array.shape[0] - 1, 4, dtype=int)

#     # Create a colormap
#     cmap = plt.cm.viridis

#     for ax, idx in zip(axes, slice_indices):
#         im = ax.imshow(array[idx, :, :], cmap=cmap)
#         ax.set_title(f"Slice {idx}")
#         ax.set_xticks([0, 32, 64, 96, 127])
#         ax.set_yticks([0, 32, 64, 96, 127])
#         ax.set_xticklabels(['X0', 'X32', 'X64', 'X96', 'X127'])
#         ax.set_yticklabels(['Y0', 'Y32', 'Y64', 'Y96', 'Y127'])

#         cbar_ax = ax.inset_axes([0.1, -0.2, 0.8, 0.05])  # [x, y, width, height]
#         cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
#         cbar.set_label('Value Scale')

#     plt.tight_layout()
#     plt.show()

# def plotter4D(listOfarrays, rows, cols):

#   # Create a 2x4 layout for 8 plots
#     fig, axes = plt.subplots(rows, cols, figsize=(20, 10), constrained_layout=False)

#     # Flatten the axes grid for easier iteration
#     axes = axes.flatten()

#     # Colormap
#     cmap = plt.cm.viridis

#     for idx, ax in enumerate(axes):
  
#         img_data = listOfarrays[idx][0, 0, :, :]
        
#         im = ax.imshow(img_data, cmap=cmap)
#         ax.set_title(f"Batch {idx}")
#         ax.set_xticks([0, 32, 64, 96, 127])
#         ax.set_yticks([0, 32, 64, 96, 127])
#         ax.set_xticklabels(['X0', 'X32', 'X64', 'X96', 'X127'])
#         ax.set_yticklabels(['Y0', 'Y32', 'Y64', 'Y96', 'Y127'])

#         # Add colorbars beneath each subplot
#         cbar_ax = ax.inset_axes([0.1, -0.2, 0.8, 0.05])  # [x, y, width, height]
#         cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
#         cbar.set_label('Value Scale')

#     plt.tight_layout()
#     plt.show()

# def plotter_batchlist(batchlist, transform=None):
#     # Create a 2x4 layout for 8 plots
#     fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=False)

#     # Flatten the axes grid for easier iteration
#     axes = axes.flatten()

#     # Colormap
#     cmap = plt.cm.viridis

#     for idx, ax in enumerate(axes):
#         # Extract the required slice [1, 1, :, :] for each batch element
#         if transform is None:
            
#             img_data = batchlist[idx][0][0][0, 1, 1, :, :]
#         else:
#             img_data = batchlist[idx][0][0][1, 1, :, :]
        
#         im = ax.imshow(img_data, cmap=cmap)
#         ax.set_title(f"Batch {idx}")
#         ax.set_xticks([0, 32, 64, 96, 127])
#         ax.set_yticks([0, 32, 64, 96, 127])
#         ax.set_xticklabels(['X0', 'X32', 'X64', 'X96', 'X127'])
#         ax.set_yticklabels(['Y0', 'Y32', 'Y64', 'Y96', 'Y127'])

#         # Add colorbars beneath each subplot
#         cbar_ax = ax.inset_axes([0.1, -0.2, 0.8, 0.05])  # [x, y, width, height]
#         cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
#         cbar.set_label('Value Scale')

#     plt.tight_layout()
#     plt.show()


# class TrainingTransform_for_rocks_Train(object):
#     # Built on Albumentations, this provides geometric transformation only  
#     def __init__(self,  prob = 1, norm = AI4BNormal_S2()):
#         self.geom_trans = A.Compose([
#                     #A.RandomCrop(width=128, height=128, p=1.0),  # Always apply random crop
#                     A.OneOf([
#                         A.HorizontalFlip(p=1),
#                         A.VerticalFlip(p=1),
#                         A.ElasticTransform(p=1), # VERY GOOD - gives perspective projection, really nice and useful - VERY SLOW   
#                         A.GridDistortion(distort_limit=0.4,p=1.),
#                         A.ShiftScaleRotate(shift_limit=0.25, scale_limit=(0.75,1.25), rotate_limit=180, p=1.0), # Most important Augmentation   
#                         ],p=1.)
#                     ],
#             additional_targets={'imageS1': 'image','mask':'mask'},
#             p = prob)
      
#         self.mytransform = self.transform_train
#         self.norm = norm
        
#     # def transform_valid(self, data):
#     #     timgS2, tmask = data
#     #     if self.norm is not None:
#     #         timgS2 = self.norm(timgS2)
        
#     #     tmask= tmask 
#     #     return timgS2,  tmask.astype(np.float32)

#     def transform_train(self, data):
#         timgS2, tmask = data
#         if self.norm is not None:
#             timgS2 = self.norm(timgS2)

#         tmask= tmask 
#         tmask = tmask.astype(np.float32)
#         # Special treatment of time series
#         c2,t,h,w = timgS2.shape
#         #print (c2,t,h,w)              
#         timgS2 = timgS2.reshape(c2*t,h,w)
#         result = self.geom_trans(image=timgS2.transpose([1,2,0]),
#                                  mask=tmask.transpose([1,2,0]))
#         timgS2_t = result['image']
#         tmask_t  = result['mask']
#         timgS2_t = timgS2_t.transpose([2,0,1])
#         tmask_t = tmask_t.transpose([2,0,1])
        
#         c2t,h2,w2 = timgS2_t.shape

        
#         timgS2_t = timgS2_t.reshape(c2,t,h2,w2)
#         return timgS2_t,  tmask_t
#     def __call__(self, *data):
#         return self.mytransform(data)

# class TrainingTransform_for_rocks_Valid(object):
#     # Built on Albumentations, this provides geometric transformation only  
#     def __init__(self, norm = AI4BNormal_S2()):
        
#         self.mytransform = self.transform_valid
#         self.norm = norm
        
#     def transform_valid(self, data):
#         timgS2, tmask = data
#         if self.norm is not None:
#             timgS2 = self.norm(timgS2)
        
#         tmask= tmask 
#         return timgS2,  tmask.astype(np.float32)
    
#     def __call__(self, *data):
#         return self.mytransform(data)
    

def checkemptyNC(pathList):
    # see if there were empty label images from the beginning
    label_index_List = []
    for i, path in enumerate(pathList):
        lb_file = xr.open_dataset(path)
        img = lb_file['band_data']
        if len(np.unique(img)) == 1:
            label_index_List.append(i)
    return label_index_List

