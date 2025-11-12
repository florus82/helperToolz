import numpy as np
import pandas as pd
from osgeo import gdal
import matplotlib.pyplot as plt
import geopandas as gpd
#import higra as hg
import os
import xarray as xr 
import osgeo
from osgeo import ogr, osr
import xml.etree.ElementTree as ET
import re
import pickle
from itertools import chain
import hashlib
from datetime import datetime, timezone
from skimage import measure
import io
import contextlib
import shutil
import sys
sys.path.append('/media/')
from helperToolz.helper import *

#####################################################################################
#####################################################################################
################## For deep leaner, mainly from feevos' repo ########################
#####################################################################################
##################################################################################### 

class AI4BNormal_S2(object):
    """
    class for Normalization of images, per channel, in format CHW 
    """
    def __init__(self):

        self._mean_s2 = np.array([5.4418573e+02, 7.6761194e+02, 7.1712860e+02, 2.8561428e+03 ]).astype(np.float32) 
        self._std_s2  = np.array( [3.7141626e+02, 3.8981952e+02, 4.7989127e+02 ,9.5173022e+02]).astype(np.float32) 

    def __call__(self,img):
        temp = img.astype(np.float32)
        temp2 = temp.T
        temp2 -= self._mean_s2
        temp2 /= self._std_s2

        temp = temp2.T
        return temp
    
def get_row_col_indices(chipsize, overlap, number_of_rows, number_of_cols):
    '''
    chipsize: the desired size of image chips passed on to GPU for prediction
    overlap: the overlap in rows and cols of image chips @chipsize
    number_of_rows, number_of_cols: overall number of rows and cols of entire datablock that should be predicted
    '''
    row_start = [i for i in range(0, number_of_rows, chipsize - overlap)]
    row_end = [i for i in range (chipsize, number_of_rows, chipsize - overlap)]
    row_start = row_start[:len(row_end)] 

    col_start = [i for i in range(0, number_of_cols, chipsize - overlap)]
    col_end = [i for i in range (chipsize, number_of_cols, chipsize - overlap)] 
    col_start = col_start[:len(col_end)]

    return [row_start, row_end, col_start, col_end]

def predict_on_GPU(path_to_model, list_of_row_col_indices, npdstack, temp_path = False):
    '''
    path_to_model: path to .pth file
    list_of_row_col_indices: a list in the order row_start, row_end, col_start, col_end (output of get_row_col_indices). This will be used to read in small chips from npdstack
    npdstack: normalized sentinel-2 npdstack (output from loadVRTintoNUmpyAI4)
    '''

    row_start = list_of_row_col_indices[0]
    row_end   = list_of_row_col_indices[1]
    col_start = list_of_row_col_indices[2]
    col_end   = list_of_row_col_indices[3]

    # define the model (.pth) and assess loss curves
    #model_name = dataFolder + 'output/models/model_state_All_but_LU_transformed_42.pth'
    model_name_short = path_to_model.split('/')[-1].split('.')[0]
 
    NClasses = 1
    nf = 96
    verbose = True
    model_config = {'in_channels': 4,
                    'spatial_size_init': (128, 128),
                    'depths': [2, 2, 5, 2],
                    'nfilters_init': nf,
                    'nheads_start': nf // 4,
                    'NClasses': NClasses,
                    'verbose': verbose,
                    'segm_act': 'sigmoid'}

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        modeli = ptavit3d_dn(**model_config).to(device)
        modeli.load_state_dict(torch.load(path_to_model))
        model = modeli.to(device) # Set model to gpu
        model.eval()
        
    preds = []

    for i in range(len(row_end)):
        for j in range(len(col_end)):
        
            image = torch.tensor(npdstack[np.newaxis, :, :, row_start[i]:row_end[i], col_start[j]:col_end[j]])
            image = image.to(torch.float)
            image = image.to(device)  # Move image to the correct device
        
            with torch.no_grad():
                pred = model(image)
                preds.append(pred.detach().cpu().numpy())
                
    torch.cuda.empty_cache()
    del model
    del modeli
    del device
    del image

    if temp_path:
        with open(path_safe(f'{temp_path}preds.pkl'), 'wb') as f:
            pickle.dump(preds, f)

    # Load again
    # with open(f'{temp_path}preds.pkl', 'rb') as f:
    #     preds = pickle.load(f)

    return preds

def export_GPU_predictions(list_of_predictions, path_to_mask, vrt_path, list_of_row_col_indices, out_path, chipsize, overlap):
    '''
    list_of_predictions: a list of predicted chips at same dimensions (output from predict_on_GPU
    path_to_mask: a path to mask that has the same dimensions as the vrt on which predictions have been undertaken; can be also a list of masks
    vrt_path: path to a folder that contains the vrt files, the predictions (and mask) is based on. Will be used for GeoTransform and Projection
    list_of_row_col_indices: a list in the order row_start, row_end, col_start, col_end (output of get_row_col_indices). 
                                Will be used to read in mask chips and manipulate Geotransform
    out_path: path to where the predicted images should be stored to
    '''

    row_start = list_of_row_col_indices[0]
    row_end   = list_of_row_col_indices[1]
    col_start = list_of_row_col_indices[2]
    col_end   = list_of_row_col_indices[3]

    if not out_path.endswith('/'):
        out_path = out_path + '/'

    gtiff_driver = gdal.GetDriverByName('GTiff')
    vrt_ds = gdal.Open(getFilelist(vrt_path, '.vrt')[0])
    geoTF = vrt_ds.GetGeoTransform()
    filenames = [f'X_{col_start[j]}_Y_{row_start[i]}.tif' for i in range(len(row_start)) for j in range(len(col_start))]

    # export unmasked chips
    for i, file in enumerate(filenames):
        out_ds = gtiff_driver.Create(path_safe(f'{out_path}unmasked_chips/chips_unmasked{str(chipsize)}_{overlap}_{file}'), int(chipsize - overlap), int(chipsize - overlap), 3, gdal.GDT_Float32)
        # change the Geotransform for each chip
        geotf = list(geoTF)
        # get column and rows from filenames
        geotf[0] = geotf[0] + geotf[1] * (int(file.split('X_')[-1].split('_')[0]) + overlap/2)
        geotf[3] = geotf[3] + geotf[5] * (int(file.split('Y_')[-1].split('.')[0]) + overlap/2)
        #print(f'X:{geoTF[0]}  Y:{geoTF[3]}  AT {file}')
        out_ds.SetGeoTransform(tuple(geotf))
        out_ds.SetProjection(vrt_ds.GetProjection())

        arr = list_of_predictions[i][0].transpose(1, 2, 0)
        for band in range(3):
            out_ds.GetRasterBand(band + 1).WriteArray(arr[int(overlap/2): -int(overlap/2), int(overlap/2): -int(overlap/2), band])
        del out_ds

    print('umnasked chips exported')

    # check if mask is a list or single mask
    if isinstance(path_to_mask, list):
        print('list of masks provided - start exporting')
        for maski in path_to_mask:
            mask_name = maski.split('cropMask_')[-1].split('.')[0]

            # load mask
            ds = gdal.Open(maski)
            mask = ds.GetRasterBand(1).ReadAsArray()

            for i, file in enumerate(filenames):
                out_ds = gtiff_driver.Create(path_safe(f'{out_path}{mask_name}/{mask_name}_{str(chipsize)}_{overlap}_{file}'), int(chipsize - overlap), int(chipsize - overlap), 3, gdal.GDT_Float32)
                # change the Geotransform for each chip
                geotf = list(geoTF)
                # get column and rows from filenames
                geotf[0] = geotf[0] + geotf[1] * (int(file.split('X_')[-1].split('_')[0]) + overlap/2)
                geotf[3] = geotf[3] + geotf[5] * (int(file.split('Y_')[-1].split('.')[0]) + overlap/2)
                #print(f'X:{geoTF[0]}  Y:{geoTF[3]}  AT {file}')
                out_ds.SetGeoTransform(tuple(geotf))
                out_ds.SetProjection(vrt_ds.GetProjection())

                arr = list_of_predictions[i][0].transpose(1, 2, 0)

                maskSub = mask[int(int(file.split('Y_')[-1].split('.')[0]) + overlap/2):chipsize + int(int(file.split('Y_')[-1].split('.')[0]) - overlap/2), 
                            int(int(file.split('X_')[-1].split('_')[0]) + overlap/2):chipsize + int(int(file.split('X_')[-1].split('_')[0]) - overlap/2)]
                for band in range(3):                
                    out_ds.GetRasterBand(band + 1).WriteArray(arr[int(overlap/2): -int(overlap/2), int(overlap/2): -int(overlap/2), band] * maskSub)
                del out_ds
             





    # # load mask
    # ds = gdal.Open(path_to_mask)
    # mask = ds.GetRasterBand(1).ReadAsArray()

    # for fold in ['chips/', 'masked_chips/']:
    #     os.makedirs(f'{out_path}{fold}', exist_ok=True)

    # for i, file in enumerate(filenames):
    #     for j in ['chips/', 'masked_chips/']:
    #         out_ds = gtiff_driver.Create(f'{out_path}{j}{str(chipsize)}_{overlap}_{file}', int(chipsize - overlap), int(chipsize - overlap), 3, gdal.GDT_Float32)
    #         # change the Geotransform for each chip
    #         geotf = list(geoTF)
    #         # get column and rows from filenames
    #         geotf[0] = geotf[0] + geotf[1] * (int(file.split('X_')[-1].split('_')[0]) + overlap/2)
    #         geotf[3] = geotf[3] + geotf[5] * (int(file.split('Y_')[-1].split('.')[0]) + overlap/2)
    #         #print(f'X:{geoTF[0]}  Y:{geoTF[3]}  AT {file}')
    #         out_ds.SetGeoTransform(tuple(geotf))
    #         out_ds.SetProjection(vrt_ds.GetProjection())

    #         arr = list_of_predictions[i][0].transpose(1, 2, 0)
    #         if j == 'masked_chips/':
    #             maskSub = mask[int(int(file.split('Y_')[-1].split('.')[0]) + overlap/2):chipsize + int(int(file.split('Y_')[-1].split('.')[0]) - overlap/2), 
    #                         int(int(file.split('X_')[-1].split('_')[0]) + overlap/2):chipsize + int(int(file.split('X_')[-1].split('_')[0]) - overlap/2)]
    #             for band in range(3):                
    #                 out_ds.GetRasterBand(band + 1).WriteArray(arr[int(overlap/2): -int(overlap/2), int(overlap/2): -int(overlap/2), band] * maskSub)
    #             del out_ds
    #         else:
    #             for band in range(3):
    #                 out_ds.GetRasterBand(band + 1).WriteArray(arr[int(overlap/2): -int(overlap/2), int(overlap/2): -int(overlap/2), band])
    #             del out_ds

def predicted_chips_to_vrt(path_to_chips, chipname, chipsize, overlap, path_to_folder_out, pyramids=False):
    '''
    path_to_chips: path to chips exported via export_GPU_predictions
    chipsize + overlap: the size of these chips (in order to select the chips if chips from different predictions are in the same folder)
    path_to_folder_out: path to FOLDER where vrt will be stored
    '''
    if not path_to_folder_out.endswith('/'):
        path_to_folder_out = path_to_folder_out + '/'
    if not path_to_chips.endswith('/'):
        path_to_chips = path_to_chips + '/'  
    path_to_chips = f'{path_to_chips}{chipname}/'
    os.makedirs(path_to_folder_out, exist_ok=True)

    chip_id = f'{chipsize}_{overlap}'
    chips = getFilelist(path_to_chips, '.tif')
    chips = [chip for chip in chips if chip_id in chip]

    # for c in chips:print(c)
    # create stacked vrts of chips
    vrt_name = f'{path_to_folder_out}{chipname}_{chipsize}_{overlap}.vrt'
    vrt = gdal.BuildVRT(vrt_name, chips, separate = False)
    vrt = None
    convertVRTpathsTOrelative(vrt_name)

    if pyramids:
        vrtPyramids(vrt_name)

def subset_mask_to_prediction_extent(path_reference_mask, path_to_prediction_vrt):
    '''
    path_reference_mask: path to the reference mask
    path_to_prediction_vrt: path to a vrt of the predicted image chips
    '''

    # check if mask has different extent from prediction
    # if so, make it the same extent for further processing (classification)
    # --> mask can never be smaller than prediciton, therefore no need to check

    ext_mask = getExtentRas(path_reference_mask)
    ext_pred = getExtentRas(path_to_prediction_vrt)

    if ext_mask == ext_pred:
        print('Mask already has same extent as prediction - no further subsetting needed :)')
    else:
        common_bounds = commonBoundsDim([ext_mask, ext_pred])
        common_coords = commonBoundsCoord(common_bounds)
        if common_bounds == ext_pred:
            ds = gdal.Open(path_reference_mask)
            in_gt = ds.GetGeoTransform()
            inv_gt = gdal.InvGeoTransform(in_gt)
            # transform coordinates into offsets (in cells) and make them integer
            off_UpperLeft = gdal.ApplyGeoTransform(inv_gt, common_coords[0]['UpperLeftXY'][0], common_coords[0]['UpperLeftXY'][1])  # new UL * rastersize^-1  + original ul/rastersize(opposite sign
            off_LowerRight = gdal.ApplyGeoTransform(inv_gt, common_coords[0]['LowerRightXY'][0], common_coords[0]['LowerRightXY'][1])
            off_ULx, off_ULy = map(round, off_UpperLeft) 
            off_LRx, off_LRy = map(round, off_LowerRight)

            band = ds.GetRasterBand(1)
            data = band.ReadAsArray(off_ULx, off_ULy, off_LRx - off_ULx, off_LRy - off_ULy)


            out_ds = gdal.GetDriverByName('GTiff').Create(path_reference_mask.split('.')[0] + '_prediction_extent.tif', 
                                                        off_LRx - off_ULx, 
                                                        off_LRy - off_ULy, 1, ds.GetRasterBand(1).DataType)
            out_gt = list(in_gt)
            out_gt[0], out_gt[3] = gdal.ApplyGeoTransform(in_gt, off_ULx, off_ULy)
            out_ds.SetGeoTransform(out_gt)
            out_ds.SetProjection(ds.GetProjection())

            out_ds.GetRasterBand(1).WriteArray(data)
            if band.GetNoDataValue():
                out_ds.GetRasterBand(1).SetNoDataValue(band.GetNoDataValue())
            del out_ds

#####################################################################################
#####################################################################################
################# S3 downloads and processing #######################################
#####################################################################################
##################################################################################### 

def getAllDatesS3(listOfFiles, year='all'):
    '''Takes a list of paths of .nc files for Sentinel-3 if year == all, all paths are considered. 
    If a year is provided, the dates are only extracted for the corresponding yearand is returned 
    as well as a pathlist subsetted to this year. 
    Expected filenaming convention: Germany_2017-01-01_2017-01-31.nc. The second 2017 is important here
    Germany_2017-1.nc works as well'''

    if year != 'all':
        listOfFiles = [file for file in listOfFiles if int(file.split('_')[-1][0:4]) == year]
    if type(listOfFiles) != list:
        listOfFiles = [listOfFiles]
    for e, file in enumerate(listOfFiles):
        dat = xr.open_dataset(file)
        if e == 0:
            tim = dat['t'].to_numpy()
        else:
            tim = np.concatenate((tim, dat['t'].to_numpy()))
    if year != 'all':
        return np.sort(tim), listOfFiles
    else:
        return np.sort(tim)

def getGeoTransFromNC(ncfile):
    '''Takes a path to an ncfile or an xarray_dataset and returns a tupel that can be used for gdal's SetGeotransform()'''
    
    if type(ncfile) == str:
        ncfile = xr.open_dataset(ncfile)
    upperLeft_X = np.min(ncfile.coords['x'])
    upperLeft_Y = np.max(ncfile.coords['y'])
    rotation = 0
    pixelWidth = ncfile.coords['x'][1] - ncfile.coords['x'][0]
    pixelHeight = -pixelWidth
    return (upperLeft_X, pixelWidth, rotation, upperLeft_Y, rotation, pixelHeight)

def getShapeFromNC(ncfile):
    '''Takes a path to an ncfile or an xarray_dataset and returns shape[1], shape[0], shape[2] ,comparable to np.array.shape()'''
    
    if type(ncfile) == str:
        ncfile = xr.open_dataset(ncfile)
    return len(ncfile.coords['x'].values), len(ncfile.coords['y'].values), ncfile[','.join(ncfile.data_vars.keys()).split(',')[-1]].shape[0]

def getDataFromNC_LST(ncfile):
    '''Takes a path to an ncfile or an xarray_dataset and returns a 3D numpy array of the data'''
    
    if type(ncfile) == str:
        ncfile = xr.open_dataset(ncfile)
    arr = ncfile[[b for b in ','.join(ncfile.data_vars.keys()).split(',') if (b == 'LST')][0]].to_numpy() # makes sure that LST exists
    return np.swapaxes(np.swapaxes(arr, 0, 1), 1, 2)

def getDataFromNC_VZA(ncfile):
    '''Takes a path to an ncfile or an xarray_dataset and returns a 3D numpy array of the data'''
    
    if type(ncfile) == str:
        ncfile = xr.open_dataset(ncfile)
    arr = ncfile[[b for b in ','.join(ncfile.data_vars.keys()).split(',') if (b == 'viewZenithAngles')][0]].to_numpy() # makes sure that LST exists
    return np.swapaxes(np.swapaxes(arr, 0, 1), 1, 2)

def getDataFromNC_VAA(ncfile):
    '''Takes a path to an ncfile or an xarray_dataset and returns a 3D numpy array of the data'''
    
    if type(ncfile) == str:
        ncfile = xr.open_dataset(ncfile)
    arr = ncfile[[b for b in ','.join(ncfile.data_vars.keys()).split(',') if (b == 'viewAzimuthAngles')][0]].to_numpy() # makes sure that LST exists
    return np.swapaxes(np.swapaxes(arr, 0, 1), 1, 2)

def getCRS_WKTfromNC(ncfile):
    '''Takes a path to an ncfile or an xarray_dataset and returns coordinate sys as wkt'''
    
    if type(ncfile) == str:
        ncfile = xr.open_dataset(ncfile)
    return ncfile['crs'].attrs['crs_wkt']

def convertNCtoTIF(ncfile, storPath, fileName, accDT, make_uint16 = False, explode=False, LST=True):
    '''Converts a filepath to an nc file or a .nc file to a .tif with option to store it UINT16 (Kelvin values are multiplied by 100 before decimals are cut off)'''
    
    gtiff_driver = gdal.GetDriverByName('GTiff')
    geoTrans = getGeoTransFromNC(ncfile)
    geoWKT = getCRS_WKTfromNC(ncfile)
    typi = gdal.GDT_Float64
    numberOfXpixels, numberOfYpixels, numberofbands = getShapeFromNC(ncfile)
    if LST:
        dat = getDataFromNC_LST(ncfile)
    else:
        dat = getDataFromNC_VZA(ncfile)
    noDataVal = np.nan

    if make_uint16 == True:
        dat = dat * 100
        dat.astype(np.uint16)
        typi = gdal.GDT_UInt16
        fileName = fileName.split('.tif')[0] + '_UINT16.tif'
        noDataVal = 0
    
    if explode == False:
        out_ds = gtiff_driver.Create(storPath + fileName, numberOfXpixels, numberOfYpixels, numberofbands, typi)
        out_ds.SetGeoTransform(geoTrans)
        out_ds.SetProjection(geoWKT)

        for band in range(numberofbands):
            out_ds.GetRasterBand(band + 1).WriteArray(dat[:, :, band])
            out_ds.GetRasterBand(band + 1).SetNoDataValue(noDataVal)
            out_ds.GetRasterBand(band + 1).SetDescription(str(accDT[band]).split('.')[0])
        del out_ds
    # explode=True --> write single raster tifs for each accDateTime
    else:
        for band in range(numberofbands):
            if make_uint16 == False:
                out_ds = gtiff_driver.Create(storPath + str(accDT[band]).split('.')[0].replace(':', '_') + '.tif', numberOfXpixels, numberOfYpixels, 1, typi)
            else:
                out_ds = gtiff_driver.Create(storPath + str(accDT[band]).split('.')[0].replace(':', '_') + '_UINT16.tif', numberOfXpixels, numberOfYpixels, 1, typi)
            out_ds.SetGeoTransform(geoTrans)
            out_ds.SetProjection(geoWKT)
            out_ds.GetRasterBand(1).WriteArray(dat[:, :, band])
            out_ds.GetRasterBand(1).SetNoDataValue(noDataVal)
            out_ds.GetRasterBand(1).SetDescription(str(accDT[band]).split('.')[0])
            del out_ds

def getAccDateTimesByfilename(dicti, filename):
    """Finds the index of a filename in lookUp['filename'] and retrieves and return corresponding accDateTimes"""
    
    if filename in dicti["filename"]:
        index = dicti["filename"].index(filename)  # Get index
        accDateTimes = dicti["accDateTimes"][index]  # Get corresponding times
        return accDateTimes
    else:
        return print(f'Filename {filename} not found!')

def exportNCarrayDerivatesInt(ncfile, storPath, fileName, bandname, arr, datType = None, numberOfBands=1, noData=None):

    gtiff_driver = gdal.GetDriverByName('GTiff')
    numberOfXpixels, numberOfYpixels, _ = getShapeFromNC(ncfile)

    if datType == None:
        typi = gdal.GDT_Float32
    else:
        typi = datType

    out_ds = gtiff_driver.Create(storPath + fileName, numberOfXpixels, numberOfYpixels, numberOfBands, typi)
    out_ds.SetGeoTransform(getGeoTransFromNC(ncfile))
    out_ds.SetProjection(getCRS_WKTfromNC(ncfile))
    if numberOfBands == 1:
        out_ds.GetRasterBand(1).WriteArray(arr)
        out_ds.GetRasterBand(1).SetDescription(bandname)
        if noData is not None:
            out_ds.GetRasterBand(1).SetNoDataValue(noData)
    else:
        for i in range(numberOfBands):
            out_ds.GetRasterBand(i+1).WriteArray(arr[:, :, i])
            out_ds.GetRasterBand(i+1).SetDescription(bandname[i])
            if noData is not None:
                out_ds.GetRasterBand(i+1).SetNoDataValue(noData)
    del out_ds

def exportNCarrayDerivatesComp(ncfile, storPath, fileName, bandnames, arr):

    gtiff_driver = gdal.GetDriverByName('GTiff')
    numberOfXpixels, numberOfYpixels, _ = getShapeFromNC(ncfile)

    out_ds = gtiff_driver.Create(storPath + fileName, numberOfXpixels, numberOfYpixels, arr.shape[2], gdal.GDT_Float32)
    out_ds.SetGeoTransform(getGeoTransFromNC(ncfile))
    out_ds.SetProjection(getCRS_WKTfromNC(ncfile))
    for i in range(arr.shape[2]):
        out_ds.GetRasterBand(i + 1).WriteArray(arr[:,:,i])
        out_ds.GetRasterBand(i + 1).SetDescription(bandnames[i])
    del out_ds

def makeGermanyMaskforNC(path_to_GER_shp, path_to_NC_file):
    '''
    makes a raster mask fitting the dimensions of an NC file
    '''
    gtiff_driver = gdal.GetDriverByName('GTiff')
    drvMemR = gdal.GetDriverByName('MEM')

    # load shapefile
    shp = ogr.Open(path_to_GER_shp, 0)
    shp_lyr = shp.GetLayer()

    numberOfXpixels, numberOfYpixels, _ = getShapeFromNC(path_to_NC_file)
    sub = drvMemR.Create('', numberOfXpixels, numberOfYpixels, 1, gdal.GDT_Int16)
    sub.SetGeoTransform(getGeoTransFromNC(path_to_NC_file))
    sub.SetProjection(getCRS_WKTfromNC(path_to_NC_file))
    band = sub.GetRasterBand(1)
    band.SetNoDataValue(0)

    gdal.RasterizeLayer(sub, [1], shp_lyr, burn_values=[1])

    return sub.ReadAsArray()

def get_pixel_size_in_target_crs(source_path, reference_path):
    '''Reprojects source raster to target SRS and returns pixel size in that SRS'''
    tmp_ds = gdal.Warp('', source_path, dstSRS=getSpatRefRas(reference_path), format='VRT')  # in-memory
    geotrans = tmp_ds.GetGeoTransform()
    x_res = geotrans[1]
    y_res = geotrans[5]
    return x_res, - y_res

def warp_raster_to_reference(source_path, reference_path, output_path, resampling='bilinear', keepRes=False, dtype=None):
    '''
    source_path: the raster to be warped
    reference_path: the raster to which will be warped
    output_path: here the warped raster will be stored; if not provided, the warped raster will be returned as memory object
    resampling: method to do resampling, e.g. bilinear, cubic, nearest
    keepRes: if set to true the warp will be done without changing the resolution of the raster at source_path to that of reference_path; if set to an integer,
    the pixel size of reference_path will be divided by that integer to gain a new pixel size 
    dtype (gdal.GDT_): if None, the dtype from source_path will be used
    '''

    # Open reference raster
    ref_ds = checkPath(reference_path)
    ref_proj = ref_ds.GetProjection()
    ref_gt = ref_ds.GetGeoTransform()
    x_size = ref_ds.RasterXSize
    y_size = ref_ds.RasterYSize

    # Extract pixel size
    ref_x_res = ref_gt[1]
    ref_y_res = -ref_gt[5]  

    # Get bounds: xmin, ymin, xmax, ymax
    xmin = ref_gt[0]
    ymax = ref_gt[3]
    xmax = xmin + ref_x_res * x_size
    ymin = ymax - ref_y_res * y_size

    # If dtype not given, use dtype from source raster
    if dtype is None:
        src_ds = checkPath(source_path)
        dtype = src_ds.GetRasterBand(1).DataType  # matches gdal.GDT_* constants
        src_ds = None  # close

    if isinstance(keepRes, bool) and keepRes:
        src_ds = checkPath(source_path)
        src_gt = src_ds.GetGeoTransform()
        src_proj = osr.SpatialReference(wkt=src_ds.GetProjection())

        ref_proj = osr.SpatialReference(wkt=ref_ds.GetProjection())

        transform = osr.CoordinateTransformation(src_proj, ref_proj)

        # Pixel corners in source CRS
        px_width = src_gt[1]
        px_height = src_gt[5]  # usually negative

        # Point (0,0)
        x0, y0, _ = transform.TransformPoint(src_gt[0], src_gt[3])
        # Point (1 pixel right)
        x1, y1, _ = transform.TransformPoint(src_gt[0] + px_width, src_gt[3])
        # Point (1 pixel down)
        x2, y2, _ = transform.TransformPoint(src_gt[0], src_gt[3] + px_height)

        # Resolution in target CRS
        aim_x_res = ((x1 - x0)**2 + (y1 - y0)**2) ** 0.5
        aim_y_res = ((x2 - x0)**2 + (y2 - y0)**2) ** 0.5

    elif isinstance(keepRes, int) and not isinstance(keepRes, bool) and keepRes > 0:
        aim_x_res = ref_x_res / keepRes
        aim_y_res = ref_y_res / keepRes

    else:
        aim_x_res = ref_x_res
        aim_y_res = ref_y_res

    out_format = 'MEM' if output_path == 'MEM' else 'GTiff'
    # Set up warp options
    warp_options = gdal.WarpOptions(
        format=out_format,
        dstSRS=ref_proj,
        outputBounds=(xmin, ymin, xmax, ymax),
        xRes=aim_x_res,
        yRes=aim_y_res,
        resampleAlg=resampling,
        targetAlignedPixels=False,
        outputType=dtype
        # srcNodata=noDat,     
        # dstNodata=-999
    )


    # Perform reprojection and resampling
    warped_ds = gdal.Warp('', source_path, options=warp_options) if out_format == 'MEM' else gdal.Warp(output_path, source_path, options=warp_options)



    # gdal.Translate(
    #     output_path,
    #     temp_path,
    #     projWin=(xmin, ymax, xmax, ymin)
    # )
    # os.remove(temp_path)

    if output_path == 'MEM':
        return warped_ds
    else:
        print(f"Raster warped and saved to: {output_path}")
         
def mask_raster(path_to_source, path_to_mask, outPath, noData=False):
    '''
    path_to_source: the raster that will be masked
    path_to_mask: the 0/1 raster mask
    outPath = masked raster will be stored here
    noData = a integer can be passed which will be set to 0
    '''
    source_ds = gdal.Open(path_to_source)
    source_band = source_ds.GetRasterBand(1)
    source_arr = source_band.ReadAsArray()

    mask_ds = gdal.Open(path_to_mask)
    mask_array = mask_ds.GetRasterBand(1).ReadAsArray()

    # Apply mask: set source data to nodata where mask == 0
    masked_array = np.where(mask_array == 0, 0, source_arr)
    if noData:
        masked_array[masked_array == noData] = 0

    masked_ds = gdal.GetDriverByName('GTiff').Create(outPath, source_ds.RasterXSize, source_ds.RasterYSize, 1, source_band.DataType)
    masked_ds.SetGeoTransform(source_ds.GetGeoTransform())
    masked_ds.SetProjection(source_ds.GetProjection())
    masked_ds.GetRasterBand(1).WriteArray(masked_array)
    masked_ds.FlushCache()

    print('masked!')

def getValsatMaxIndex(arr1, arr2):
    """ This functions extracts values from one array at those indices, where another array has its maximum in values (axis =2),
    and will return them as np.array(2D)

    Args:
        arr1 (np.array(3D)): is the array, where the maximum of values are computed across axis=2
        arr2 (np.array(3D)): is the array from which values will be extracted at those indices where arr1 has its maxima 
    """
    # first, take care of np.nan as it messes up the results

    arr_no_nan = np.where(np.isnan(arr1), -np.inf, arr1)
    idx = np.argmax(arr_no_nan, axis = 2)
    rows, cols = np.indices(arr1.shape[0:2])

    return arr2[rows, cols, idx]

#####################################################################################
#####################################################################################
################# Preprocess FORCE Output ###########################################
#####################################################################################
##################################################################################### 

def sortListwithOtherlist(list1, list2, rev=False):
    ''' list1: unsorted list
        list2: unsorted list with same length as list1
        Sorts list2 based on sorted(list1). Returns sorted list1 list2
        if rev == True, list will be returned reversed
        '''
    sortlist1, sortlist2 = zip(*sorted(zip(list1, list2)))
    sort1 = list(sortlist1)
    sort2 = list(sortlist2)

    if rev:
        sort1.reverse()
        sort2.reverse()
  
    return sort1, sort2

# def getBluGrnRedBnrFORCEList(filelist):
#     '''Takes a list of paths to an exploded FORCE output and returns a list with ordered paths
#     First all blue then green, red and bnir bands. Furthermore, paths are chronologically sorted (1,2,3,4..months)'''
#     blu = [file for file in filelist if file.split('SEN2H_')[-1].split('_')[0] == 'BLU']
#     grn = [file for file in filelist if file.split('SEN2H_')[-1].split('_')[0] == 'GRN']
#     red = [file for file in filelist if file.split('SEN2H_')[-1].split('_')[0] == 'RED']
#     bnr = [file for file in filelist if file.split('SEN2H_')[-1].split('_')[0] == 'BNR']

#     blu = sortListwithOtherlist([int(t.split('-')[-1].split('.')[0]) for t in blu], blu)[-1]
#     grn = sortListwithOtherlist([int(t.split('-')[-1].split('.')[0]) for t in grn], grn)[-1]
#     red = sortListwithOtherlist([int(t.split('-')[-1].split('.')[0]) for t in red], red)[-1]
#     bnr = sortListwithOtherlist([int(t.split('-')[-1].split('.')[0]) for t in bnr], bnr)[-1]

#     return sum([blu, grn, red, bnr], [])

def getCOLORSinOrderFORCELIST(filelist, colorlist, single=False):
    """Takes a list of paths to exploded FORCE files, that contain a 3 lettered word, e.g. BLU, GRN, and returns a list of chronological ordered paths (per color)
    e.g. blu1,blu2,blu3,grn1,grn2,grn2

    Args:
        filelist (list): list of strings paths to FORCE files (exploded)
        colorlist (list): list of colors found in FORCE output, e.g. BLU, GRN, BNR
        single (bool): if True, a nested list with single color lists will be returned instead of ordered single list
    """
    conti = []
    for color in colorlist:
        pattern = fr'_{color.upper()}_' 
        conti.append([file for file in filelist if re.search(pattern, file)])
    if single:
        return list(chain.from_iterable(conti))
    else:
        return conti

def getFORCExyRangeName(tiles):
    '''take a list of subsetted FORCE Tile names in the Form of X0069_Y0042 and returns a string to be used as filename 
    that gives X and Y range ,e.g. Force_X_from_68_to_69_Y_from_42_to_42'''
    X = [int(tile.split('_')[0][-2:]) for tile in tiles]
    Y = [int(tile.split('_')[1][-2:]) for tile in tiles]

    return f'Force_X_from_{min(X)}_to_{max(X)}_Y_from_{min(Y)}_to_{max(Y)}'

def convertVRTpathsTOrelative(vrt_path):
    tree = ET.parse(vrt_path)
    root = tree.getroot()

    for source in root.findall(".//SourceFilename"):
        abs_path = source.text
        rel_path = os.path.relpath(abs_path, os.path.dirname(vrt_path))  # Convert to relative
        source.text = rel_path
        source.set("relativeToVRT", "1")  # Add the attribute

    # Save the modified VRT file
    tree.write(vrt_path)

def vrtPyramids(vrtpath):
    '''takes a vrtpath (or gdalOpened vrt) and produces pyramids'''
    if type(vrtpath) == osgeo.gdal.Dataset:
        image = vrtpath
    else:
        Image = gdal.Open(vrtpath, 0) # 0 = read-only, 1 = read-write. 
    gdal.SetConfigOption('COMPRESS_OVERVIEW', 'DEFLATE')
    Image.BuildOverviews("NEAREST", [2,4,8,16,32,64])
    del Image

def reduce_forceTSA_output_to_validmonths(path_to_forceoutput, start_month_int, end_month_int):
    '''path_to_forceoutput: path of stored force output (quite likely you want the folder in which all tile folders are)
    start_month_int & end_month_int: e.g. 3 for march and 8 for August
    Please note: The filter will look for the YYYYMMDD characters that come right before .tif
    '''
    # get rid of force output that is not needed -> months outside of growing season that do not exist in AI4Boundaries
    files = getFilelist(path_to_forceoutput, '.tif', deep=True)

    filesToKill = [f for f in files if int(f.split('-')[-1].split('.')[0]) not in [i for i in range(start_month_int, end_month_int + 1, 1)]]

    for file in filesToKill:
        RasterKiller(file)

    return list(filter(lambda item: item not in filesToKill, files))

def reduce_forceTSI_output_to_validmonths(path_to_forceoutput, start_month_int, end_month_int):
    '''path_to_forceoutput: path of stored force output (quite likely you want the folder in which all tile folders are)
    start_month_int & end_month_int: e.g. 3 for march and 8 for August
    Please note: The filter will look for the YYYYMMDD characters that come right before .tif
    '''
    # get rid of force output that is not needed -> months outside of growing season that do not exist in AI4Boundaries
    files = getFilelist(path_to_forceoutput, '.tif', deep=True)

    #filesToKill = [f for f in files if int(f.split('-')[-1].split('.')[0]) not in [i for i in range(start_month_int, end_month_int + 1, 1)]]
    filesToKill = [f for f in files if int(re.search(r'(\d{4})(\d{2})(\d{2})\.tif$', f)[2]) not in [i for i in range(start_month_int, end_month_int + 1, 1)]]

    for file in filesToKill:
        RasterKiller(file)

    return list(filter(lambda item: item not in filesToKill, files))

def get_forceTSI_output_DOYS(listOfFORCEoutput):
    '''
    Will return a sorted list of unique DOYs in format YYYYMMDD. Please note, that this only works with files in FORCE naming convention, where
    the date will be used that is at the end of the filename (YYYYMMDD.tif)
    listOfFORCEoutput: list with paths to tif files from FORCE TSI output
    '''
    return sorted(list(set([re.search(r'(\d{4})(\d{2})(\d{2})\.tif$', file)[0].split('.tif')[0] for file in listOfFORCEoutput])))

def get_forceTSI_output_Tiles(listOfFORCEoutput):
    """
    Will return a sorted list of unique Tiles Ids(e.g. 'X0057_Y0044')

    Args:
        listOfFORCEoutput (list_of_strings): list with paths to tif files from FORCE TSI output
    """
    return sorted(list(set([re.search(r'X\d{4}_Y\d{4}',file)[0] for file in listOfFORCEoutput])))

def check_forceTSI_compositionDates(listOfFORCEoutput):
    """_summary_

    Args:
        listOfFORCEoutput (list_of_strings): list with paths to tif files from FORCE TSI output
    """
    fatal_check = 0
    date_list = []
    tiles = get_forceTSI_output_Tiles(listOfFORCEoutput)
    for tile in tiles:
        date_list.append((get_forceTSI_output_DOYS([file for file in listOfFORCEoutput if tile in file])))
    for i in range(0,len(date_list)-1):
        if date_list[i] == date_list[i + 1]:
            continue
        else:
            fatal_check = 1
    if fatal_check:
        print('the doys of composites across tiles is not equal - Better check!!!!! - No date list returned!!!!!!')
    else:
        print('all dates of composites are the same :)')
        return date_list[0]
    
def createFORCEtileLIST(xlist, ylist, withDash=False):
    """
    Create a list, which entries resemble FORCE tile IDs in the format 'X00xx_Y00yy'
    Args:
        x_list (list of integer): each x tile coordinate will be paired with the y tile coordinate at the same index at ylist
        ylist (list of integer): each y tile coordinate will be paired with the x tile coordinate at the same index at xlist
        e.g. xlist[0] = 10 and ylist[0] = 20 --> 'X0010_Y0020'
        withDash (boolean): if TRUE, X0000_Y0000
    """
    if withDash:
        return [f'X{x:04d}_Y{y:04d}' for x, y in zip(xlist, ylist)]
    else:
        return [f'X{x:04d}Y{y:04d}' for x, y in zip(xlist, ylist)]

def get_forcetiles_range(list_of_forcefiles):
    '''list_of_forcefiles: e.g. output from reduce_force_to_validmonths
    creats a string that indicates X and Y extremes from list_of_forcefiles'''
    return list(set([re.search(r'X\d{4}_Y\d{4}', tile).group() for tile in list_of_forcefiles]))

def force_order_Colors_for_VRT(list_of_forcefiles, list_of_colors, list_of_days):
    '''list_of_forcefiles: e.g. output from reduce_force_to_validmonths
        will return a list that orders the input list to blue, green, red, ir independently from tiles and dates'''
    tiles = list(set([file.split('output/')[-1].split('/')[2].split('/')[0] for file in list_of_forcefiles]))
    tilefilesL = []
    for color in list_of_colors:
        for day in list_of_days:
            for tile in tiles:
                for file in list_of_forcefiles:
                    if tile in file and color in file and day in file and tile in file:
                        tilefilesL.append(file)


    if len(tilefilesL) == len(list_of_colors) * len(list_of_days) * len(tiles):
        return [tilefilesL[i:i+len(tiles)] for i in range(0, len(tilefilesL), len(tiles))]
    else:
        raise ValueError('length of colors, days and files do not line up')

def force_to_vrt(list_of_forcefiles, ordered_forcetiles, vrt_out_path, pyramids=False, bandnames=False):
    '''list_of_forcefiles: e.g. output from reduce_force_to_validmonths
        ordered_forcetiles: e.g output from getCOLORSinOrderFORCELIST (single=False)
        vrt_out_path: path where .vrt files will be created (there will be more than one to account for all the bands)
        pyramids: if set to True, pyramids will be created (might be very very large!!)'''
    
    # tiles = list(set([file.split('output/')[-1].split('/')[1].split('/')[0] for file in list_of_forcefiles]))
    force_folder_name = getFORCExyRangeName(get_forcetiles_range(list_of_forcefiles))
    if not vrt_out_path.endswith('/'):
        vrt_out_path = vrt_out_path + '/'
    outDir = f'{vrt_out_path}{force_folder_name}/'
    if not os.path.exists(outDir):
        os.makedirs(outDir)
        print(outDir)
        vrts = []
        for i in range(len(ordered_forcetiles)):
            vrt_name = f'{outDir}{force_folder_name}_{str(i)}.vrt'
            vrt = gdal.BuildVRT(vrt_name, ordered_forcetiles[i], separate = False)
            vrt = None

            # make paths in vrts relative
            convertVRTpathsTOrelative(vrt_name)
            vrts.append(vrt_name)

        # set optionally bandnames    
        if bandnames:
            for idz, bname in enumerate(np.repeat(bandnames,int(len(ordered_forcetiles) / len(bandnames))).tolist()):  
                print(f'{outDir}{force_folder_name}_{str(idz)}.vrt')
                vrt = gdal.Open(f'{outDir}{force_folder_name}_{str(idz)}.vrt', gdal.GA_Update)  # VRT must be writable
                band = vrt.GetRasterBand(1)
                band.SetDescription(bname)
                vrt = None
        print('single vrts created')
        
        nums = [int(vrt.split('_')[-1].split('.')[0]) for vrt in vrts]
        vrts_sorted = sortListwithOtherlist(nums, vrts)[-1]
        print('paths in vrts made relative')
        
        vrt = gdal.BuildVRT(f'{outDir}{force_folder_name}_Cube.vrt', vrts_sorted, separate = True)
        vrt = None
        if bandnames:
            # set vrt band names
            vrt = gdal.Open(f'{outDir}{force_folder_name}_Cube.vrt', gdal.GA_Update)  # VRT must be writable
            for idz, bname in enumerate(np.repeat(bandnames,int(len(ordered_forcetiles) / len(bandnames))).tolist()): 
                band = vrt.GetRasterBand(1+idz)
                band.SetDescription(bname)
            vrt = None
        # convertVRTpathsTOrelative(f'{outDir}{force_folder_name}_Cube.vrt')
        print('overlord vrt created')
        if pyramids:
            # build pyramids
            vrtPyramids(f'{outDir}{force_folder_name}_Cube.vrt')
            print('VRT created with pyramids')
    else:
        print('Vrt might already exist - please check!!')

def loadVRTintoNumpyAI4(vrtPath, applyNormalizer=True):
    '''vrtPath: path in which vrts are stored
        vrts will be loaded into numpy array and normalized (for Sentinel-2 10m bands!!!!!)'''
    vrtFiles = [file for file in getFilelist(vrtPath, '.vrt') if 'Cube' not in file]
    vrtFiles = sortListwithOtherlist([int(vrt.split('_')[-1].split('.')[0]) for vrt in vrtFiles], vrtFiles)[-1]
    bands = []

    for vrt in vrtFiles:
        ds = gdal.Open(vrt)
        bands.append(ds.GetRasterBand(1).ReadAsArray())
    cube = np.dstack(bands)
   
    data_cube = np.transpose(cube, (2, 0, 1))
    reshaped_cube = data_cube.reshape(4, 6, ds.RasterYSize, ds.RasterXSize)
    normalizer = AI4BNormal_S2()
    if applyNormalizer:
        return normalizer(reshaped_cube)
    else:
        return reshaped_cube

#####################################################################################
#####################################################################################
################# 05_Prepare_Validation_and_Validation ##############################
#####################################################################################
##################################################################################### 

########### sub-functions
def export_intermediate_products(row_col_start, intermediate_aray, dummy_gt, dummy_proj, folder_out, filename, noData=None, typ='int', comp=False):
    '''
    intermediate_aray: array to be exported
    dummy_gt + dummy_proj: GetGeotransform() and GetProjection from a gdal.Open object that contains desired geoinformation
    folder_out: path to FOLDER, where intermediate product will be stored
    noData = a no data value can be assigned to the exported tif
    comp (bool): If True, tiff uses options=['COMPRESS=DEFLATE', 'TILED=YES']
    '''
    if not folder_out.endswith('/'):
        folder_out = folder_out + '/'

    row_start = int(row_col_start.split('_')[0])
    col_start = int(row_col_start.split('_')[1])
    
    typi = gdal.GDT_Int32
    if typ == 'float':
        typi = gdal.GDT_Float32
    if comp:
        out_ds = gdal.GetDriverByName('GTiff').Create(f'{folder_out}{filename}', 
                                                    intermediate_aray.shape[1], intermediate_aray.shape[0], 1, typi,
                                                    options=['COMPRESS=DEFLATE', 'TILED=YES'])
    else:    
        out_ds = gdal.GetDriverByName('GTiff').Create(f'{folder_out}{filename}', 
                                                    intermediate_aray.shape[1], intermediate_aray.shape[0], 1, typi)
    # change the Geotransform for each chip
    geotf = list(dummy_gt)
    # get column and rows from filenames
    geotf[0] = geotf[0] + geotf[1] * col_start
    geotf[3] = geotf[3] + geotf[5] * row_start
    #print(f'X:{geoTF[0]}  Y:{geoTF[3]}  AT {file}')
    out_ds.SetGeoTransform(tuple(geotf))
    out_ds.SetProjection(dummy_proj)
                
    out_ds.GetRasterBand(1).WriteArray(intermediate_aray)
    if noData != None:
        out_ds.GetRasterBand(1).SetNoDataValue(noData)
    del out_ds

def makeTif_np_to_matching_tif(array, tif_path, path_to_file_out, noData = None, gdalType = None, bands=1):
    '''
    exports an np.array to a tif, based on a tif that has the same extent. Probably, the np.array is a manipulation of that tif
    array: the numpy array
    tif_path: path to the tif from which geoinformation will be extracted
    path_to_file_out: where the new tif should be stored
    noData = a no data value can be assigned to the exported tif
    gdaType = a different data type can be set here, otherwise, the one from hte tif at tif_path will be used
    bands = default single band raster, provide the integer of bands to export as stack
    '''
    ds = gdal.Open(tif_path)
    gtiff_driver = gdal.GetDriverByName('GTiff')
    no_data = ds.GetRasterBand(1).GetNoDataValue()
    if gdalType == None:
        dtypi = ds.GetRasterBand(1).DataType
    else:
        dtypi = gdalType

    out_ds = gtiff_driver.Create(path_to_file_out, ds.RasterXSize, ds.RasterYSize, bands, dtypi)
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())
    if bands == 1:
        out_ds.GetRasterBand(1).WriteArray(array)
        if noData != None:
            out_ds.GetRasterBand(1).SetNoDataValue(noData)
        else:
            if no_data is not None:
                out_ds.GetRasterBand(1).SetNoDataValue(no_data)
    else:
        for b in range(bands):
            out_ds.GetRasterBand(b + 1).WriteArray(array[:,:,b])
        if noData != None:
            for b in range(bands):
                out_ds.GetRasterBand(b + 1).SetNoDataValue(noData)
        else:
            if no_data is not None:
                out_ds.GetRasterBand(b + 1).SetNoDataValue(no_data)

    del out_ds

def makePyramidsForTif(tif_path):
    ds = gdal.Open(tif_path)
    ds.BuildOverviews("AVERAGE", [2, 4, 8, 16, 32])
    ds = None
    print('pyramids created')

def TooCloseToBorder(numbered_array, border_limit):
    rows, cols = np.where(numbered_array==True)
    r,c = numbered_array.shape
    if any(value < border_limit for value in [np.min(rows), r - np.max(rows), np.min(cols), c - np.max(cols)]):
        return True
    
def InstSegm(extent, boundary, t_ext=0.4, t_bound=0.2):
    """
    INPUTS:
    extent : extent prediction
    boundary : boundary prediction
    t_ext : threshold for extent
    t_bound : threshold for boundary
    OUTPUT:
    instances
    """

    # Threshold extent mask
    ext_binary = np.uint8(extent >= t_ext)

    # Artificially create strong boundaries for
    # pixels with non-field labels
    input_hws = np.copy(boundary)
    input_hws[ext_binary == 0] = 1

    # Create the directed graph
    size = input_hws.shape[:2]
    graph = hg.get_8_adjacency_graph(size)
    edge_weights = hg.weight_graph(
        graph,
        input_hws,
        hg.WeightFunction.mean
    )

    tree, altitudes = hg.watershed_hierarchy_by_dynamics(
        graph,
        edge_weights
    )
    
    # Get individual fields
    # by cutting the graph using altitude
    instances = hg.labelisation_horizontal_cut_from_threshold(
        tree,
        altitudes,
        threshold=t_bound)
    
    instances[ext_binary == 0] = -1

    return instances


def get_IoUs(row_col_start, extent_true, extent_pred, boundary_pred, t_ext, 
             t_bound, dummy_gt, dummy_proj, intermediate_path, intermediate=True):
    
    # get predicted instance segmentation
    instances_pred = InstSegm(extent_pred, boundary_pred, t_ext=t_ext, t_bound=t_bound)
    instances_pred = measure.label(instances_pred, background=-1) 
    if intermediate:# and row_col_start == '10760_17982':
            export_intermediate_products(row_col_start, instances_pred, dummy_gt, dummy_proj,\
                                        intermediate_path, filename=f'{t_ext}_{t_bound}_instance_pred_{row_col_start}.tif', noData=0)

    # get instances from ground truth label; already done globally during joblist creation
    # binary_true = extent_true > 0
    # instances_true = measure.label(binary_true, background=0, connectivity=1)
    instances_true = extent_true
    if intermediate:# and row_col_start == '10760_17982':
            export_intermediate_products(row_col_start, instances_true, dummy_gt, dummy_proj,\
                                        intermediate_path, filename=f'{t_ext}_{t_bound}_instance_true_{row_col_start}.tif', noData=0)

    # loop through true fields
    field_values = np.unique(instances_true)
    
    best_IoUs = []
    field_IDs = []
    field_sizes = []
    pred_field_overlap = []
    centroid_rows = []
    centroid_cols = []
    centroid_IoUS = []
    centroid_IDs = []
    intersect_IDs  = []

    for field_value in field_values: # loops over the sampled IACS poylgons
        if field_value == 0:
            continue # move on to next value

        this_field = instances_true == field_value # makes a binary raster for the respective sampled IACS poylgon
        this_field_centroid = np.mean(np.column_stack(np.where(this_field)),axis=0).astype(int) # calculates the centroid of that polygon
        

        # fill lists with info
        centroid_rows.append(this_field_centroid[0])
        centroid_cols.append(this_field_centroid[1])
        field_IDs.append(field_value)
        field_sizes.append(np.sum(this_field))
        
        # find predicted fields that intersect with true field
        intersecting_fields = this_field * instances_pred # multiplies binary raster of sampled IACS poylgon with prediction --> only overlapping predicted fields in raster
        intersect_values = np.unique(intersecting_fields) # get the labeled IDS from intersecting predicted fields
  
        # compute IoU for each intersecting field
        field_IoUs = []
        intersect_area = []
        centroid_IoU = 0
        centroid_ID = 0

        for intersect_value in intersect_values:
            if intersect_value == 0:
                field_IoUs.append(0)
                intersect_area.append(0)
                continue # move on to next value
            
            pred_field = instances_pred == intersect_value # makes a binary raster of of intersecting predicted field
            pred_field_area = np.sum(pred_field) # calculates the area of that polygon

            # calculate IoU
            union = this_field + pred_field > 0
            intersection = (this_field * pred_field) > 0
            IoU = np.sum(intersection) / np.sum(union)
            field_IoUs.append(IoU)
            intersect_area.append(np.sum(intersection) / pred_field_area)

            # check for centroid condition
            if instances_pred[this_field_centroid[0], this_field_centroid[1]] == intersect_value:
                centroid_IoU = IoU
                centroid_ID = intersect_value
    
        # take maximum IoU - this is the IoU for this true field
        if len(field_IoUs) > 1 or field_IoUs[0] != 0:
            best_IoUs.append(np.max(field_IoUs))
            pred_field_overlap.append(intersect_area[np.argmax(field_IoUs)])
            intersect_IDs.append(intersect_values[np.argmax(field_IoUs)])
        else:
            best_IoUs.append(0)
            pred_field_overlap.append(0)
            intersect_IDs.append(0)
        
        # fill centroid list
        centroid_IoUS.append(centroid_IoU)
        centroid_IDs.append(centroid_ID)
    

    # export centroids and intersecting fields with best IoUs
    if intermediate:# and row_col_start == '10760_17982':

            # Create mask of intersecting fields with best IoUs
            intersect_mask = np.isin(instances_pred, centroid_IDs)# intersectL)
            filtered_instances_pred = instances_pred * intersect_mask
            
            # centroids
            for r,c, cid in zip(centroid_rows, centroid_cols, centroid_IDs):
                filtered_instances_pred[r, c] = cid

                export_intermediate_products(row_col_start, filtered_instances_pred, dummy_gt, dummy_proj, \
                                            intermediate_path, filename=f'{t_ext}_{t_bound}_intersected_at_max_and_centroids_{row_col_start}.tif', noData=0)

    return best_IoUs, centroid_IoUS, centroid_rows, centroid_cols, centroid_IDs, field_IDs, field_sizes, intersect_IDs, pred_field_overlap

########## main-function

def get_IoUs_per_Tile(row_col_start, extent_true, extent_pred, boundary_pred, result_dir, \
                      dummy_gt, dummy_proj, intermediate_path, intermediate=True, t_ext=False, t_bound=False):
    
    print(f'Starting on tile {row_col_start} for {result_dir}')
    # make a dictionary for export
    k = ['row_col_start','t_ext','t_bound', 'max_IoU', 'centroid_IoU', 'centroid_row', 'centroid_col',\
          'centroid_IDs', 'reference_field_IDs', 'reference_field_sizes', 'intersect_IDs', 'intersect_area'] #'medianIoU', 'meanIoU', 'IoU_50', 'IoU_80']
    v = [list() for i in range(len(k))]
    res = dict(zip(k, v))

    # set the parameter combinations and test combinations
    if not t_ext:
        t_exts = [i/100 for i in range(10,95,5)] 
        t_bounds = [i/100 for i in range(10,95,5)]
    else:
        if isinstance(t_ext, list):
            t_exts = t_ext
            t_bounds = t_bound
        else:
            t_exts = [t_ext]
            t_bounds = [t_bound]
    # loop over parameter combinations
    for t_ext in t_exts:
        for t_bound in t_bounds:
            #print('thresholds: ' + str(t_ext) + ', ' +str(t_bound))

            img_IoUs, centroid_IoUS, centroid_rows, centroid_cols, centroid_IDs, field_IDs, field_sizes , intersect_IDS, intersect_area = \
                get_IoUs(row_col_start, extent_true, extent_pred, boundary_pred, t_ext, t_bound, dummy_gt, \
                         dummy_proj, intermediate_path, intermediate=intermediate)
            
            for e, IoUs in enumerate(img_IoUs):
    
                res['row_col_start'].append(row_col_start)
                res['t_ext'].append(t_ext)
                res['t_bound'].append(t_bound)
                res['max_IoU'].append(IoUs)
                res['centroid_IoU'].append(centroid_IoUS[e])
                res['centroid_row'].append(centroid_rows[e])
                res['centroid_col'].append(centroid_cols[e])
                res['centroid_IDs'].append(centroid_IDs[e])
                res['reference_field_IDs'].append(field_IDs[e])
                res['reference_field_sizes'].append(field_sizes[e])
                res['intersect_IDs'].append(intersect_IDS[e])
                res['intersect_area'].append(intersect_area[e])
    
    # export results
    df  = pd.DataFrame(data = res)
    df.to_csv(f'{result_dir}/{row_col_start}_IoU_hyperparameter_tuning.csv', index=False)

    print(f'Finished tile {row_col_start}')

#####################################################################################
#####################################################################################
################# General stuff #####################################################
#####################################################################################
##################################################################################### 

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

def plotter(array, row=1, col=1, names=False, title=False):

    # Plot the slices
    fig, axes = plt.subplots(row, col, figsize=(col*5, row*5), constrained_layout=False)  # 4 slices
    # Create a colormap
    cmap = plt.cm.viridis
    
    if col != 1:
        slice_indices = np.linspace(0, (row * col) -1, col * row, dtype=int)
        #print(slice_indices)
        for ax, idx in zip(axes.ravel(), slice_indices):
            im = ax.imshow(array[:, :, idx], cmap=cmap)
            if names == False:
                ax.set_title(f"Slice {idx}")
            else:
                ax.set_title(names[idx], fontsize=10)     
            # ax.set_xticks([0, 32, 64, 96, 127])
            # ax.set_yticks([0, 32, 64, 96, 127])
            # ax.set_xticklabels(['X0', 'X32', 'X64', 'X96', 'X127'])
            # ax.set_yticklabels(['Y0', 'Y32', 'Y64', 'Y96', 'Y127'])

            cbar_ax = ax.inset_axes([0.1, -0.2, 0.8, 0.05])  # [x, y, width, height]
            cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
            cbar.set_label('Value Scale')
    else:
            im = axes.imshow(array[:, :], cmap=cmap)
            #ax.set_xticks([0, 32, 64, 96, 127])
            #ax.set_yticks([0, 32, 64, 96, 127])
            #ax.set_xticklabels(['X0', 'X32', 'X64', 'X96', 'X127'])
            #ax.set_yticklabels(['Y0', 'Y32', 'Y64', 'Y96', 'Y127'])

            cbar_ax = axes.inset_axes([0.1, -0.2, 0.8, 0.05])  # [x, y, width, height]
            cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
            cbar.set_label('Value Scale')
 
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    if title != False:
        fig.suptitle(f'Date is {title[0]} at canals {title[1]}', fontsize=12)
    plt.tight_layout()
    plt.show()

def plotter_side_by_side(arr1, arr2, titles=("Array 1", "Array 2"), cmap="viridis"):
    """
    Plots two arrays next to each other.

    Parameters:
        arr1, arr2 : arrays of the same shape
        titles : tuple of strings, titles for each subplot
        cmap : colormap for the images
    """
    if arr1.shape != arr2.shape:
        raise ValueError("Arrays must have the same shape")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im0 = axes[0].imshow(arr1, cmap=cmap)
    axes[0].set_title(titles[0])
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    im1 = axes[1].imshow(arr2, cmap=cmap)
    axes[1].set_title(titles[1])
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

def getNestedListMinLengthIndex(nestedList):
    res = [index for index, band in enumerate(nestedList) if len(band) == min([len(i) for i in nestedList])]
    return res[0]

def getBandNames(rasterstack):
    bands = []
    ds = gdal.Open(rasterstack)
    numberBands = ds.RasterCount
    for i in range(numberBands):
        bands.append(ds.GetRasterBand(i+1).GetDescription())
    return bands

def makeZeroNAN(arr):
    arr[arr == 0] = np.nan
    return arr

def RasterKiller(raster_path):
    if os.path.isfile(raster_path):
        os.remove(raster_path)

def getAttributesName(layer):

    # check the type of layer
    if type(layer) is ogr.Layer:
        lyr = layer

    elif type(layer) is ogr.DataSource:
        lyr = layer.GetLayer(0)

    elif type(layer) is str:
        lyrOpen = ogr.Open(layer)
        lyr = lyrOpen.GetLayer(0)

    # create empty dict and fill it
    header = dict.fromkeys(['Name', 'Type'])
    head   = [[lyr.GetLayerDefn().GetFieldDefn(n).GetName(),
             ogr.GetFieldTypeName(lyr.GetLayerDefn().GetFieldDefn(n).GetType())]
            for n in range(lyr.GetLayerDefn().GetFieldCount())]

    header['Name'], header['Type'] = zip(*head)

    return header

def getAttributesALL(layer):

    # check the type of layer
    if type(layer) is ogr.Layer:
        lyr = layer

    elif type(layer) is ogr.DataSource:
        lyr = layer.GetLayer(0)

    elif type(layer) is str:
        lyrOpen = ogr.Open(layer)
        lyr = lyrOpen.GetLayer(0)

    # create empty dict and fill it

    header = dict.fromkeys(['Name', 'Type'])

    head = [[lyr.GetLayerDefn().GetFieldDefn(n).GetName(),
             ogr.GetFieldTypeName(lyr.GetLayerDefn().GetFieldDefn(n).GetType())]
            for n in range(lyr.GetLayerDefn().GetFieldCount())]

    header['Name'], header['Type'] = zip(*head)

    attrib = dict.fromkeys(header['Name'])
    for i, j in enumerate(header['Name']):
        attrib[j] = [lyr.GetFeature(k).GetField(j) for k in range(lyr.GetFeatureCount())]

    return attrib

def getSpatRefRas(layer):
    # check type of layer
    if type(layer) is gdal.Dataset:
        SPRef = osr.SpatialReference()
        SPRef.ImportFromWkt(layer.GetProjection())

    elif type(layer) is str:
        lyr   = gdal.Open(layer)
        SPRef = osr.SpatialReference()
        SPRef.ImportFromWkt(lyr.GetProjection())

    #print(SPRef)
    return(SPRef)

def getSpatRefVec(layer):

    # check the type of layer
    if type(layer) is ogr.Geometry:
        SPRef   = layer.GetSpatialReference()

    elif type(layer) is ogr.Feature:
        lyrRef  = layer.GetGeometryRef()
        SPRef   = lyrRef.GetSpatialReference()

    elif type(layer) is ogr.Layer:
        SPRef   = layer.GetSpatialRef()

    elif type(layer) is ogr.DataSource:
        lyr     = layer.GetLayer(0)
        SPRef   = lyr.GetSpatialRef()

    elif type(layer) is str:
        lyrOpen = ogr.Open(layer)
        lyr     = lyrOpen.GetLayer(0)
        SPRef   = lyr.GetSpatialRef()

    return(SPRef)

def getExtentRas(raster):
    if type(raster) is str:
        ds = gdal.Open(raster)
    elif type(raster) is gdal.Dataset:
        ds = raster
    gt = ds.GetGeoTransform()
    ext = {'Xmin': gt[0],
            'Xmax': gt[0] + (gt[1] * ds.RasterXSize),
            'Ymin': gt[3] + (gt[5] * ds.RasterYSize),
            'Ymax': gt[3]}
    return ext

def commonBoundsDim(extentList):
    # create empty dictionary with list slots for corner coordinates
    k = ['Xmin', 'Xmax', 'Ymin', 'Ymax']
    v = [[], [], [], []]
    res = dict(zip(k, v))

    # fill it with values of all raster files
    for i in extentList:
        for j in k:
            res[j].append(i[j])
    # determine min or max values per values' list to get common bounding box
    ff = [max, min, max, min]
    for i, j in enumerate(ff):
        res[k[i]] = j(res[k[i]])
    return res

def commonBoundsCoord(ext):
    if type(ext) is dict:
        ext = [ext]
    else:
        ext = ext
    cooL = []
    for i in ext:
        coo = {'UpperLeftXY': [i['Xmin'], i['Ymax']],
               'UpperRightXY': [i['Xmax'], i['Ymax']],
               'LowerRightXY': [i['Xmax'], i['Ymin']],
               'LowerLeftXY': [i['Xmin'], i['Ymin']]}
        cooL.append(coo)
    return cooL

def list_to_uniques(input_list):
    '''
    return unique values of input_list
    '''
    return list(dict.fromkeys(input_list))

def is_leap_year(year):
    return (year % 4 == 0) and (year % 100 != 0 or year % 400 == 0)

def reprojShapeEPSG(file, epsg):
    # create spatial reference object
    sref  = osr.SpatialReference()
    sref.ImportFromEPSG(epsg)
    # open the shapefile
    ds = ogr.Open(file, 1)
    driv = ogr.GetDriverByName('ESRI Shapefile')  # will select the driver foir our shp-file creation.

    shapeStor = driv.CreateDataSource('/'.join(file.split('/')[:-1]))
    # get first layer (assuming ESRI is standard) & and create empty output layer with spatial reference plus object type
    in_lyr = ds.GetLayer()
    out_lyr = shapeStor.CreateLayer(file.split('/')[-1].split('.')[0] + '_reproj_' + str(epsg), sref, in_lyr.GetGeomType())

# create attribute field
    out_lyr.CreateFields(in_lyr.schema)
    # with attributes characteristics
    out_feat = ogr.Feature(out_lyr.GetLayerDefn())

    for in_feat in in_lyr:
        geom = in_feat.geometry().Clone()
        geom.TransformTo(sref)
        out_feat.SetGeometry(geom)
        for i in range(in_feat.GetFieldCount()):
            out_feat.SetField(i, in_feat.GetField(i))
        out_lyr.CreateFeature(out_feat)
    shapeStor.Destroy()
    del ds
    return('reprojShape done :)')

def get_UTM_zone_and_corners_from_xml(nc_file_path, xml_file_path_list):
    '''
    relatively hard coded
    nc_file_path: for this nc file, the extracted general xml will be found
    xml_file_path_list: here will the mattching xml file be searched
    '''
    base_name = nc_file_path.rsplit(".", maxsplit=1)[0]  # remove .nc extension
    f_xml = [f for f in xml_file_path_list if f.startswith(base_name) and f.endswith(("T1.xml", "T2.xml"))][0]

    tree = ET.parse(f_xml)
    root = tree.getroot()

    # define namespace
    ns = {"espa": "http://espa.cr.usgs.gov/v2"}
    utm = root.find(f'.//espa:zone_code', ns).text
    ul_corner = root.find(
        ".//espa:projection_information/espa:corner_point[@location='UL']", namespaces=ns
    )
    # Extract X and Y values
    ul_x = float(ul_corner.attrib["x"])
    ul_y = float(ul_corner.attrib["y"])

    lr_corner = root.find(
        ".//espa:projection_information/espa:corner_point[@location='LR']", namespaces=ns
    )
    # Extract X and Y values
    lr_x = float(lr_corner.attrib["x"])
    lr_y = float(lr_corner.attrib["y"])

    return [utm, ul_x, ul_y, lr_x, lr_y]

def stackReader(path_to_stack, bands=False, era=False):
    """Reads-in a raster stacks and returns a 3D numpy array of that array.
    Optionally, a list with band names will be returned

    Args:
        path_to_stack (str): path to the stack.tif
        bands (bool): If True, a list with band names of stack wil be returned as well. (WORKS ONLY WITH ERA5.grib files for now!!!!)
        era (bool): If bands True and era True, the bands metadata is extracted in a different manner
    """
    conti = []
    if type(path_to_stack) != osgeo.gdal.Dataset:
        ds = gdal.Open(path_to_stack)
    else:
        ds = path_to_stack
    ds = checkPath(path_to_stack)
    bandCount = ds.RasterCount
    if bands:
        bandsL = []
        if bandCount > 1:
            for b in range(bandCount):
                conti.append(ds.GetRasterBand(b+1).ReadAsArray())
                if not era:
                    bandsL.append(ds.GetRasterBand(b+1).GetDescription())
                else:
                    bandsL.append(datetime.fromtimestamp(int(ds.GetRasterBand(b+1).GetMetadata()['GRIB_VALID_TIME']), tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'))
            return np.dstack(conti), bandsL
        else:
            conti.append(ds.GetRasterBand(1).ReadAsArray())
            if not era:
                bandsL.append(ds.GetRasterBand(1).GetDescription())
            else:
                bandsL.append(datetime.fromtimestamp(int(ds.GetRasterBand(1).GetMetadata()['GRIB_VALID_TIME']), tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'))
            return np.dstack(conti), bandsL
    else:
        if bandCount > 1:
            for b in range(bandCount):
                conti.append(ds.GetRasterBand(b+1).ReadAsArray())
            return np.dstack(conti)
        else:
            return ds.GetRasterBand(1).ReadAsArray()

def stack_tifs(input_tif_list, output_tif=False, d_type=False):
    # Open the first raster to get geotransform, projection, and shape
    if type(input_tif_list) != osgeo.gdal.Dataset:
        src0 = gdal.Open(input_tif_list[0])
    else:
        src0 = input_tif_list
    x_size = src0.RasterXSize
    y_size = src0.RasterYSize
    proj = src0.GetProjection()
    geotrans = src0.GetGeoTransform()
    if d_type:
        dtype = d_type
    else:
        dtype = src0.GetRasterBand(1).DataType
    num_bands = len(input_tif_list)

    # Create output multi-band raster
    if output_tif:
        out_ds = gdal.GetDriverByName('GTiff').Create(output_tif, x_size, y_size, num_bands, dtype)
    else:
        out_ds = gdal.GetDriverByName('MEM').Create('', x_size, y_size, num_bands, dtype)
    out_ds.SetProjection(proj)
    out_ds.SetGeoTransform(geotrans)

    # Write each input raster as a band
    for i, tif_path in enumerate(input_tif_list):
        src = gdal.Open(tif_path)
        band_data = src.GetRasterBand(1).ReadAsArray()
        out_ds.GetRasterBand(i + 1).WriteArray(band_data)

    if output_tif:
        out_ds.FlushCache()
        out_ds = None  # Close file 
    else:
        return out_ds

def npTOdisk(arr, reference_path, outPath, bands = False, bandnames = False, noData = False, d_type = False):
    """exports a numpy array to a tif that is stored on disk

    Args:
        arr (numpy array): the array to be exported
        reference_path (str): path to the reference tif. The extent and dimensions must fit!!!!
        outPath (_str): path to exported tif on disk
    """
    ref_ds = checkPath(reference_path)
    ref_band = ref_ds.GetRasterBand(1)
    if not bands:
        bands = ref_ds.RasterCount
    if not d_type:
        out_ds = gdal.GetDriverByName('GTiff').Create(outPath, ref_ds.RasterXSize, ref_ds.RasterYSize, bands, ref_band.DataType)
    else:
        out_ds = gdal.GetDriverByName('GTiff').Create(outPath, ref_ds.RasterXSize, ref_ds.RasterYSize, bands, d_type)
    out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
    out_ds.SetProjection(ref_ds.GetProjection())
    if bands == 1:
        out_ds.GetRasterBand(1).WriteArray(arr)
        if bandnames:
            out_ds.GetRasterBand(1).SetDescription(bandnames)
        if noData:
            out_ds.GetRasterBand(1).SetNoDataValue(noData)
    else:
        for i in range(bands):
            out_ds.GetRasterBand(i+1).WriteArray(arr[:,:,i])
            if bandnames:
                out_ds.GetRasterBand(i+1).SetDescription(str(bandnames[i]))
            if noData:
                out_ds.GetRasterBand(i+1).SetNoDataValue(noData)
    out_ds.FlushCache()

def checkPath(path):
    if isinstance(path, str):
            return gdal.Open(path)
    else:
        return path
    
def getUniqueIDfromTILESXY(tileXlist, tileYlist):
    s = ",".join(map(str, tileXlist + tileYlist))
    return hashlib.sha256(s.encode()).hexdigest()

def maskVRT(vrtPath, maskArray):
    """Opens a vrt and masks it with a binary array of the same dimensions

    Args:
        vrtPath (str): path to the vrt
        maskArray (np.array): binary np array, where 1 == valid and 0 == maks
    """
    ds = gdal.Open(vrtPath)
    b = []
    for band in range(ds.RasterCount):
        b.append(ds.GetRasterBand(band + 1).ReadAsArray() * maskArray)
    masked_arr =  np.dstack(b)
    makeTif_np_to_matching_tif(masked_arr, vrtPath, f"{vrtPath.split('.')[0]}.tif", 0, bands=len(b))

def maskVRT(vrtPath, maskArray, suffix):
    """Opens a vrt and masks it with a binary array of the same dimensions

    Args:
        vrtPath (str): path to the vrt
        maskArray (np.array): binary np array, where 1 == valid and 0 == maks
    """
    ds = gdal.Open(vrtPath)
    b = []
    for band in range(ds.RasterCount):
        b.append(ds.GetRasterBand(band + 1).ReadAsArray() * maskArray)
    masked_arr =  np.dstack(b)
    makeTif_np_to_matching_tif(masked_arr, vrtPath, f"{vrtPath.split('.')[0]}{suffix}.tif", bands=len(b))

def maskVRT_water(vrtPath):
    """Opens a vrt and applies dirty water mask --> slope = NA and aspect = 180

    Args:
        vrtPath (str): path to the vrt
        maskArray (np.array): binary np array, where 1 == valid and 0 == maks
    """
    ds = gdal.Open(vrtPath)
    b = []
    for band in range(ds.RasterCount):
        b.append(ds.GetRasterBand(band + 1).ReadAsArray())
    arr =  np.dstack(b)
    mask = np.logical_and(arr[:,:,10] < 0.000000001, arr[:,:,11] == 180)
    masked_arr = np.where(mask[:,:,None],np.nan, arr)
    makeTif_np_to_matching_tif(masked_arr, vrtPath, f"{vrtPath.split('.')[0]}_watermask.tif", gdalType=gdal.GDT_Float32, bands=len(b))

def maskVRT_water_and_drop_aux(vrtPath):
    """Opens a vrt and applies dirty water mask --> slope = NA and aspect = 180
    after masking, slope, aspect, incidence will be dropped

    Args:
        vrtPath (str): path to the vrt
        maskArray (np.array): binary np array, where 1 == valid and 0 == maks
    """
    ds = gdal.Open(vrtPath)
    b = []
    for band in range(ds.RasterCount):
        b.append(ds.GetRasterBand(band + 1).ReadAsArray())
    arr =  np.dstack(b)
    mask = np.logical_and(arr[:,:,10] < 0.000000001, arr[:,:,11] == 180)
    masked_arr = np.where(mask[:,:,None],np.nan, arr)
    masked_arr = masked_arr[:,:,0:10]
    makeTif_np_to_matching_tif(masked_arr, vrtPath, f"{vrtPath.split('.')[0]}_watermask.tif", gdalType=gdal.GDT_Float32, bands=9)


def silent(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)
    
def path_safe(path):
    """when storing a file, this function makes sure that the directory exists, where file will be stored

    Args:
        path (str): path to file (or directory)
    """
    if os.path.splitext(path)[1]: # checks if path points to a file
            dir_path = os.path.dirname(path)
    else:
        dir_path = path  # treat as directory

    if dir_path == "":
        print('this is not a path!!!')
        return path

    os.makedirs(dir_path, exist_ok=True)
    return path

def dirfinder(path):
    """ returns a list with all directory names within a folder

    Args:
        path (str): str path to folder that will be searched for directories

    Returns:
        list: list of directory names (str)
    """
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def getAttributesALL(path_to_vector):
    """
    Returns a dictionary of all attributes (fields) and their values
    from a path to a .shp, gpkg, .parquet ...
    """

    gdf = gpd.read_file(path_to_vector)

    return {col: gdf[col].tolist() for col in gdf.columns if col != 'geometry'}

def clear_directory(path):
    """
    Delete all files and folders inside path, but keep the directory itself.
    """

    confirm = input(f"Are you sure you want to delete everythin @ '{path}'? (y/N): ").strip().lower()
    if confirm not in ("y", "yes"):
        print("Operation cancelled.")
        return
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                # Delete files or symlinks
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                # Recursively delete directories
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")