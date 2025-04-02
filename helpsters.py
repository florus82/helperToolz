import numpy as np
import pandas as pd
from osgeo import gdal
import matplotlib.pyplot as plt
import math
import os
import time
import xarray as xr 
import osgeo
from osgeo import ogr, osr
import random
import xml.etree.ElementTree as ET


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

#####################################################################################
#####################################################################################
################# S3 downloads to numpy/tiff files ##################################
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


def getDataFromNC(ncfile):
    '''Takes a path to an ncfile or an xarray_dataset and returns a 3D numpy array of the data'''
    
    if type(ncfile) == str:
        ncfile = xr.open_dataset(ncfile)
    arr = ncfile[[b for b in ','.join(ncfile.data_vars.keys()).split(',') if b == 'LST'][0]].to_numpy() # makes sure that LST exists
    return np.swapaxes(np.swapaxes(arr, 0, 1), 1, 2)
    

def getCRS_WKTfromNC(ncfile):
    '''Takes a path to an ncfile or an xarray_dataset and returns coordinate sys as wkt'''
    
    if type(ncfile) == str:
        ncfile = xr.open_dataset(ncfile)
    return ncfile['crs'].attrs['crs_wkt']


def convertNCtoTIF(ncfile, storPath, fileName, accDT, make_uint16 = False, explode=False):
    '''Converts a filepath to an nc file or a .nc file to a .tif with option to store it UINT16 (Kelvin values are multiplied by 100 before decimals are cut off)'''
    
    gtiff_driver = gdal.GetDriverByName('GTiff')
    geoTrans = getGeoTransFromNC(ncfile)
    geoWKT = getCRS_WKTfromNC(ncfile)
    typi = gdal.GDT_Float64
    numberOfXpixels, numberOfYpixels, numberofbands = getShapeFromNC(ncfile)
    dat = getDataFromNC(ncfile)
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


def exportNCarrayDerivatesInt(ncfile, storPath, fileName, bandname, arr, make_uint16 = False):

    gtiff_driver = gdal.GetDriverByName('GTiff')
    numberOfXpixels, numberOfYpixels, numberofbands = getShapeFromNC(ncfile)

    typi = gdal.GDT_Float32
    if make_uint16 == True:
        typi = gdal.GDT_Int16

    out_ds = gtiff_driver.Create(storPath + fileName, numberOfXpixels, numberOfYpixels, 1, typi)
    out_ds.SetGeoTransform(getGeoTransFromNC(ncfile))
    out_ds.SetProjection(getCRS_WKTfromNC(ncfile))
    out_ds.GetRasterBand(1).WriteArray(arr)

    out_ds.GetRasterBand(1).SetDescription(bandname)
    del out_ds


#####################################################################################
#####################################################################################
################# Preprocess FORCE Output ###########################################
#####################################################################################
##################################################################################### 

def sortListwithOtherlist(list1, list2):
    ''' list1: unsorted list
        list2: unsorted list with same length as list1
        Sorts list2 based on sorted(list1). Returns sorted list1 list2'''
    sortlist1, sortlist2 = zip(*sorted(zip(list1, list2)))
    return list(sortlist1), list(sortlist2)


def getBluGrnRedBnrFORCEList(filelist):
    '''Takes a list of paths to an exploded FORCE output and returns a list with ordered paths
    First all bluem then green, red and bnir bands. Furthermore, paths are chronologically sorted (1,2,3,4..months)'''
    blu = [file for file in filelist if file.split('SEN2H_')[-1].split('_')[0] == 'BLU']
    grn = [file for file in filelist if file.split('SEN2H_')[-1].split('_')[0] == 'GRN']
    red = [file for file in filelist if file.split('SEN2H_')[-1].split('_')[0] == 'RED']
    bnr = [file for file in filelist if file.split('SEN2H_')[-1].split('_')[0] == 'BNR']

    blu = sortListwithOtherlist([int(t.split('-')[-1].split('.')[0]) for t in blu], blu)[-1]
    grn = sortListwithOtherlist([int(t.split('-')[-1].split('.')[0]) for t in grn], grn)[-1]
    red = sortListwithOtherlist([int(t.split('-')[-1].split('.')[0]) for t in red], red)[-1]
    bnr = sortListwithOtherlist([int(t.split('-')[-1].split('.')[0]) for t in bnr], bnr)[-1]

    return sum([blu, grn, red, bnr], [])


def getFORCExyRange(tiles):
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


def sortListwithOtherlist(list1, list2):
    ''' list1: unsorted list
        list2: unsorted list with same length as list1
        Sorts list2 based on sorted(list1). Returns sorted list1 list2'''
    sortlist1, sortlist2 = zip(*sorted(zip(list1, list2)))
    return list(sortlist1), list(sortlist2)


def vrtPyramids(vrtpath):
    '''takes a vrtpath (or gdalOpened vrt) and produces pyramids'''
    if type(vrtpath) == osgeo.gdal.Dataset:
        image = vrtpath
    else:
        Image = gdal.Open(vrtpath, 0) # 0 = read-only, 1 = read-write. 
    gdal.SetConfigOption('COMPRESS_OVERVIEW', 'DEFLATE')
    Image.BuildOverviews("NEAREST", [2,4,8,16,32,64])
    del Image


#####################################################################################
#####################################################################################
################# General stuff #####################################################
#####################################################################################
##################################################################################### 

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
