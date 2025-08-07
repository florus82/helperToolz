from helperToolz.helpsters import is_leap_year
from helperToolz.dicts_and_lists import *
import numpy as np
from osgeo import gdal
from datetime import datetime, timezone
import re
import pandas as pd
import pvlib
from pvlib.irradiance import get_total_irradiance
from pvlib.location import Location

def transform_compositeDate_into_LSTbands(compDate, dayrange):
    """calculates which LST bands are associated with a compDate, based on dayrange
    A dictionary will be returned that contains the information on the month and band of each LST to be read-in
    Please note, that this benefits LST daily composites stored as monthly stacks, where the band of each stack corresponds to the day of that month

    Args:
        compDate (string): the date from the FORCE filesname in string in format YYYYMMDD
        dayrange (int): LST bands from how many days +/- from compdate should be considered
    """
    year = int(compDate[0:4])

    # what bands (days) in LST from which month are associated with the current S2 scene
    if is_leap_year(year):
        dayCountList = DAYCOUNT_LEAP
        end_day = 367
        add_to_april = 1 # needed because LST from +/-4 days away2 from S2 compos
    else:
        dayCountList = DAYCOUNT_NOLEAP
        end_day = 366
        add_to_april = 0

    day_sequence = [f'{year}{month:02d}{day:02d}' 
            for month, days_in_month in enumerate(dayCountList, start=1)
            for day in range(1, days_in_month + 1)]
        
    dicti = {}

    for i in range(1, end_day):
        month_idx = np.where(np.cumsum(dayCountList) >= i)[0][0]
        day_in_month =  i - (np.cumsum(dayCountList)[month_idx - 1] if month_idx > 0 else 0)

        dicti[i] = {
            'month': INT_TO_MONTH[f'{(month_idx + 1):02d}'],
            'band': int(day_in_month)
        }

    index_in_sequence = [i+1 for i, day in enumerate(day_sequence) if day == compDate][0]

    # check for the special case, when compDate is YYYY0406 and dayrange == 4 as with this setting 1 April would not be processed
    # due to the FORCE processing with TSI at 9 day intervall that happened
    if (add_to_april == 1 and index_in_sequence == 97) or (add_to_april == 0 and index_in_sequence == 96):
        dayrange_low = dayrange + 1
        dayrange_up = dayrange
    else:
        dayrange_low = dayrange
        dayrange_up = dayrange

    # check for the special case, when compDate is YYYY1027 and dayrange == 4 as with this setting November would be processed
    if (add_to_april == 1 and index_in_sequence >= 302) or (add_to_april == 0 and index_in_sequence >= 301):
        dayrange_low = dayrange
        dayrange_up = (add_to_april + 304 - index_in_sequence)
    else:
        dayrange_low = dayrange
        dayrange_up = dayrange


    return {k: v for k, v in dicti.items() if index_in_sequence - dayrange_low <= k <= index_in_sequence + dayrange_up}

def growingSeasonChecker(month, lower_month=4, upper_month=10):
    """Checks if a month is within range of set months, Returns bool

    Args:
        month (int): integer representation of month, e.g. 3 == March
        lower_month (integer): integer for first month that will return True
        upper_month (_type_): integer of last month that will return True
    """
    month = INT_TO_MONTH[f'{month:02d}']
    if month in [INT_TO_MONTH[f'{i:02d}'] for i in range(lower_month, upper_month + 1)]:
        return True
    else:
        return False

def warp_ERA5_to_reference(grib_path, reference_path, output_path, resampling='bilinear', NoData=False, single=False,
                           sharp_DEM=None, sharp_geopot=None, sharp_rate=None, sharp_blendheight=None):
    """Warps a ERA .gib file to an existing raster in terms of resolution, projection, extent and aligns to raster.
    Optionally, the output can be DEM sharpened to a blending height (if all parameters are provided) before export

    Args:
        grib_path (str): complete path to the grib file
        reference_path (str): complete path to the DEM raster to which the grib dataset will be warped to
        output_path (str): path to filename where the raster stacks will be stored. If output_path == 'MEM', the warped raster will be returned in memory
        resampling (str, optional): _description_. Defaults to 'bilinear'.
        bands (list_of_int, optional): Defaults to 'ALL'. A subset of bands needs to be provided as a list of integer values
        NoData (int, optional): if not provided, the function will try to retrieve the NoData value from the dataset
        single (bool): if set to True, only the first band of ERA will be exported (--> geopotential)

    """
    # Open datasets
    ref_ds = gdal.Open(reference_path)
    src_ds = gdal.Open(grib_path)

    if NoData:
        nodat = NoData
    else:
        try:
            nodat = src_ds[list(src_ds.data_vars.keys())[0]].attrs['GRIB_missingValue']
        except:
            nodat = -9999

    # Get geotransform and projection from reference
    gt = ref_ds.GetGeoTransform()
    proj = ref_ds.GetProjection()

    # Calculate bounds
    xmin = gt[0]
    ymax = gt[3]
    xres = gt[1]
    yres = -gt[5]
    xmax = xmin + ref_ds.RasterXSize * xres
    ymin = ymax - ref_ds.RasterYSize * yres

    warp_options = gdal.WarpOptions(
        format='MEM',
        dstSRS=proj,
        xRes=xres,
        yRes=yres,
        outputBounds=(xmin, ymin, xmax, ymax),
        resampleAlg=resampling,
        dstNodata=nodat,
        # targetAlignedPixels=True
    )

    # Perform reprojection and resampling
    warped_ds = gdal.Warp('', grib_path, options=warp_options)

    if all([sharp_DEM, sharp_geopot, sharp_blendheight, sharp_rate]):
        print('DEM Sharpener will be applied')
         # load sharpener datasets
        dem_ds = gdal.Open(sharp_DEM)
        dem = dem_ds.GetRasterBand(1).ReadAsArray()
        geopot_ds = gdal.Open(sharp_geopot)
        geopot = geopot_ds.GetRasterBand(1).ReadAsArray()
        
        if not single:
            bandCount = warped_ds.RasterCount
            if output_path != 'MEM':
                out_ds = gdal.GetDriverByName('GTiff').Create(output_path,
                                                                warped_ds.RasterXSize,
                                                                warped_ds.RasterYSize,
                                                                bandCount, gdal.GDT_Float32)
                out_ds.SetGeoTransform(warped_ds.GetGeoTransform())
                out_ds.SetProjection(warped_ds.GetProjection())

                for i in range(bandCount):
                    band = warped_ds.GetRasterBand(i+1)
                    bandname = datetime.fromtimestamp(int(band.GetMetadata()['GRIB_VALID_TIME']), tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                    in_arr = band.ReadAsArray()
                    out_ds.GetRasterBand(i+1).WriteArray(applyAdiabaticDEMsharpener(in_arr,
                                                                                    dem,
                                                                                    geopot,
                                                                                    sharp_rate,
                                                                                    sharp_blendheight))
                    out_ds.GetRasterBand(i+1).SetNoDataValue(nodat)
                    out_ds.GetRasterBand(i+1).SetDescription(bandname)
                del out_ds
            else:
                out_ds = gdal.GetDriverByName('MEM').Create('',
                                                                warped_ds.RasterXSize,
                                                                warped_ds.RasterYSize,
                                                                bandCount, gdal.GDT_Float32)
                out_ds.SetGeoTransform(warped_ds.GetGeoTransform())
                out_ds.SetProjection(warped_ds.GetProjection())

                for i in range(bandCount):
                    band = warped_ds.GetRasterBand(i+1)
                    bandname = datetime.fromtimestamp(int(band.GetMetadata()['GRIB_VALID_TIME']), tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                    in_arr = band.ReadAsArray()
                    out_ds.GetRasterBand(i+1).WriteArray(applyAdiabaticDEMsharpener(in_arr,
                                                                                    dem,
                                                                                    geopot,
                                                                                    sharp_rate,
                                                                                    sharp_blendheight))
                    out_ds.GetRasterBand(i+1).SetNoDataValue(nodat)
                    out_ds.GetRasterBand(i+1).SetDescription(bandname)
                return out_ds

        else:
            if output_path != 'MEM':
                out_ds = gdal.GetDriverByName('GTiff').Create(output_path,
                                                                warped_ds.RasterXSize,
                                                                warped_ds.RasterYSize,
                                                                1, gdal.GDT_Float32)
                out_ds.SetGeoTransform(warped_ds.GetGeoTransform())
                out_ds.SetProjection(warped_ds.GetProjection())
                band = warped_ds.GetRasterBand(1)
                bandname = datetime.fromtimestamp(int(band.GetMetadata()['GRIB_VALID_TIME']), tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                in_arr = band.ReadAsArray()
                out_ds.GetRasterBand(1).WriteArray(applyAdiabaticDEMsharpener(in_arr,
                                                                            dem,
                                                                            geopot,
                                                                            sharp_rate,
                                                                            sharp_blendheight))
                out_ds.GetRasterBand(1).SetNoDataValue(nodat)
                out_ds.GetRasterBand(1).SetDescription(bandname)
                del out_ds
            else:
                out_ds = gdal.GetDriverByName('MEM').Create('',
                                                warped_ds.RasterXSize,
                                                warped_ds.RasterYSize,
                                                1, gdal.GDT_Float32)
                out_ds.SetGeoTransform(warped_ds.GetGeoTransform())
                out_ds.SetProjection(warped_ds.GetProjection())
                band = warped_ds.GetRasterBand(1)
                bandname = datetime.fromtimestamp(int(band.GetMetadata()['GRIB_VALID_TIME']), tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                in_arr = band.ReadAsArray()
                out_ds.GetRasterBand(1).WriteArray(applyAdiabaticDEMsharpener(in_arr,
                                                                            dem,
                                                                            geopot,
                                                                            sharp_rate,
                                                                            sharp_blendheight))
                out_ds.GetRasterBand(1).SetNoDataValue(nodat)
                out_ds.GetRasterBand(1).SetDescription(bandname)

    elif any([sharp_DEM, sharp_geopot, sharp_blendheight, sharp_rate]):
        raise ValueError('Not all datasets for sharpening provided - Please provide all or none sharpener parameter')
    
    else:
        if not single:
            bandCount = warped_ds.RasterCount

            if output_path != 'MEM':
                out_ds = gdal.GetDriverByName('GTiff').Create(output_path,
                                                                warped_ds.RasterXSize,
                                                                warped_ds.RasterYSize,
                                                                bandCount, gdal.GDT_Float32)
                out_ds.SetGeoTransform(warped_ds.GetGeoTransform())
                out_ds.SetProjection(warped_ds.GetProjection())

                for i in range(bandCount):
                    band = warped_ds.GetRasterBand(i+1)
                    bandname = datetime.fromtimestamp(int(band.GetMetadata()['GRIB_VALID_TIME']), tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                    in_arr = band.ReadAsArray()
                    out_ds.GetRasterBand(i+1).WriteArray(in_arr)
                    out_ds.GetRasterBand(i+1).SetNoDataValue(nodat)
                    out_ds.GetRasterBand(i+1).SetDescription(bandname)
                del out_ds
            else:
                out_ds = gdal.GetDriverByName('MEM').Create('',
                                            warped_ds.RasterXSize,
                                            warped_ds.RasterYSize,
                                            bandCount, gdal.GDT_Float32)
                out_ds.SetGeoTransform(warped_ds.GetGeoTransform())
                out_ds.SetProjection(warped_ds.GetProjection())

                for i in range(bandCount):
                    band = warped_ds.GetRasterBand(i+1)
                    bandname = datetime.fromtimestamp(int(band.GetMetadata()['GRIB_VALID_TIME']), tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                    in_arr = band.ReadAsArray()
                    out_ds.GetRasterBand(i+1).WriteArray(in_arr)
                    out_ds.GetRasterBand(i+1).SetNoDataValue(nodat)
                    out_ds.GetRasterBand(i+1).SetDescription(bandname)
                
                return out_ds
        
        else:
            if output_path != 'MEM':
                out_ds = gdal.GetDriverByName('GTiff').Create(output_path,
                                                                warped_ds.RasterXSize,
                                                                warped_ds.RasterYSize,
                                                                1, gdal.GDT_Float32)
                out_ds.SetGeoTransform(warped_ds.GetGeoTransform())
                out_ds.SetProjection(warped_ds.GetProjection())
                band = warped_ds.GetRasterBand(1)
                bandname = datetime.fromtimestamp(int(band.GetMetadata()['GRIB_VALID_TIME']), tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                in_arr = band.ReadAsArray()
                out_ds.GetRasterBand(1).WriteArray(in_arr)
                out_ds.GetRasterBand(1).SetNoDataValue(nodat)
                out_ds.GetRasterBand(1).SetDescription(bandname)
                del out_ds
            else:
                return warped_ds

def applyAdiabaticDEMsharpener(era5, dem, geopot, rate, bheight):
    """takes a era5 value at blending height and converts to surface temperature using a dem and a lapse rate
    follows the formula from Mohammad
    Args:
        era5 (np_array_float): contains era5 variable at 100 meter
        dem (np_array_float): contains DEM values
        geopot (np_array_float): contains geopotential values
        rate (float): the lapse rate, e.g. constant DRY_ADIABAT
        bheight (int): the assumed heigth at which era5 was recorded
    """
    return era5 - rate * ((dem + bheight) - (2 + (geopot/GRAVITY)))

# def applyDEMsharpener(era5, dem, geopot, bheight):
#     """takes a era5 value at blending height and converts to surface temperature using a dem
#     adapted the formula from Mohammad
#     Args:
#         era5 (np_array_float): contains era5 variable at 100 meter
#         dem (np_array_float): contains DEM values
#         geopot (np_array_float): contains geopotential values
#         bheight (int): the assumed heigth at which era5 was recorded
#     """
#     return era5 - rate * ((dem + bheight) - (2 + (geopot/GRAVITY)))

def find_grib_file(grib_path_list, lst_acquisition_file):
    """Returns the string(s) to path of grib that matches the year and month of LST acquisition file and variable(s) within the list

    Args:
        path_to_grib (list): list of strings --> full paths to grib files (multiple variables can be provided)
        lst_acquisition_file (str): the year and month will be extracted from this filename, so that grib file will match it 
    """
    matchpathL = []
    year = re.search(r'\b20\d{2}\b', lst_acquisition_file).group()
    for yfile in [file for file in grib_path_list if str(year) in file]:
        month = int(yfile.split(f'{year}_')[-1].split('.')[0])
        if INT_TO_MONTH[f'{int(month):02d}'] == lst_acquisition_file.split('_')[-1].split('.tif')[0]:
            matchpathL.append(yfile)
    return matchpathL

def get_warped_ERA5_at_doy(path_to_era_grib, lst_acq_file, doy, outPath='MEM', sharp_blendheight=False, sharp_DEM=False, sharp_geopot=False, sharp_rate=False):
    """will get the era5 value (interpolated between 2 neighbouring values) at the time of LST acquisition per pixel for one LST composite (1 day)
    Returns a 2D numpy array
    Args:
        path_to_era_grib (str): full path to grib file (monthly)
        lst_acq_file (str): file to acquisition LST file (format: ...../filename_year_month.tif)
        doy (int): the day of lst_acq_file that should be processed --> lst_acq_file has one band per day
        output_path (str): path to filename where the raster stacks will be stored. If output_path == 'MEM', the warped raster will be returned in memory
        sharp_blendheight (int): Defaults to False. If sharpening should be applied, all 4 parameters must be provided
        sharp_DEM (str): _description_. Path to DEM.
        sharp_geopot (str): Path to geopotential file
        sharp_rate (constant): adiabatic rate to be chosen from constants
        output_path (str): path to filename where the raster stacks will be stored. If output_path == 'MEM', the warped raster will be returned in memory
    """
    if outPath == 'MEM':
        era_ds = warp_ERA5_to_reference(path_to_era_grib, lst_acq_file, outPath, sharp_blendheight=sharp_blendheight, sharp_DEM=sharp_DEM, sharp_geopot=sharp_geopot, sharp_rate=sharp_rate)
    else:
        warp_ERA5_to_reference(path_to_era_grib, lst_acq_file, outPath, sharp_blendheight=sharp_blendheight, sharp_DEM=sharp_DEM, sharp_geopot=sharp_geopot, sharp_rate=sharp_rate)
        era_ds = gdal.Open(outPath)
    bandNumber = era_ds.RasterCount
    era_time = [pd.Timestamp(era_ds.GetRasterBand(i+1).GetDescription()) for i in range(bandNumber)]

    # open and load the acquisition raster from LST
    LST_acq_ds = gdal.Open(lst_acq_file)
    arr = LST_acq_ds.GetRasterBand(doy).ReadAsArray()
    arr = arr.astype(np.int64)
    arr_flat = arr.ravel()
    arr_ts = pd.to_datetime(arr_flat, unit='s')
    arr_up = pd.Series(arr_ts).dt.ceil('h')
    arr_up = arr_up.values.reshape(arr.shape)
    arr_min = pd.Series(arr_ts).dt.minute
    arr_min = arr_min.values.reshape(arr.shape)

    # get the relevant bands of era5 dataset (i.e. those bands, that reflect the upper and lower )
    bands = []
    for d1 in np.unique(arr_up):
        for count, e5 in enumerate(era_time):
            if d1 == e5:
                bands.append(count)

    # load the era5 variable acquisition-wise into 2D numpy array 
    arrL = []
    for b in bands:
        # compare the time of LST composite with 
        mask = arr_up == era_time[b]
        # load the band that holds the observation just after LST acquisition
        after = era_ds.GetRasterBand(b).ReadAsArray()
        # load the band that holds the observation just prior to LST acquisition
        before = era_ds.GetRasterBand(b-1).ReadAsArray()
        # calculate the linearly interpolated values at the minute of acquisition
        vals = before - (before - after) * (np.array(arr_min, dtype=np.float16) / 60)
        arrL.append(vals * mask)
    block = np.dstack(arrL)

    return np.nanmax(block, axis = 2)

def warp_np_to_reference(arr, arr_tif_path, target_tif_path, noData=np.nan, resamp=gdal.GRA_Bilinear, output_path=None):
    """
    Warps a NumPy array to the spatial resolution, projection, and extent of a target tif. Therefore, a path to a tif that holds the geoinfo of the array
    must be provided. Returns the warped array, while export as tif is optional.

    Args:
        arr (numeric np.array): hols the data that should be warped
        arr_tif_path (str): path to the tif that holds the geoinformation of arr, on which basis the array will be warped
        Target_tif_path (_type_): the tif, to which properties the array will be warped
        noData (numeric): Value of array that represents the noData value. Defaults to np.nan.
        resamp (gdal.GRA_, optional): Algorythm that should be used for resampling. Defaults to gdal.GRA_Bilinear.
        output_path (str, optional): If provided, the warped array will be stored as tiff at this location. Defaults to None.
    """
    # determine whether arr has more than one band  
    arrDim = len(arr.shape)

    # open the tif file associated with the array and extract geo metadata
    src_ds = gdal.Open(arr_tif_path)
    gt_src = src_ds.GetGeoTransform()
    proj_src = src_ds.GetProjection()
    cols_src = src_ds.RasterXSize
    rows_src = src_ds.RasterYSize

    # create an in-memory tif of array to warp it
    mem_drv = gdal.GetDriverByName('MEM')
    if arrDim == 2:
        src_mem = mem_drv.Create('', cols_src, rows_src, 1, gdal.GDT_Float32)
        src_mem.SetGeoTransform(gt_src)
        src_mem.SetProjection(proj_src)
        src_mem.GetRasterBand(1).WriteArray(arr)
        src_mem.GetRasterBand(1).SetNoDataValue(noData)
    else:
        src_mem = mem_drv.Create('', cols_src, rows_src, arr.shape[2], gdal.GDT_Float32)
        src_mem.SetGeoTransform(gt_src)
        src_mem.SetProjection(proj_src)
        for band in range(arr.shape[2]):
            src_mem.GetRasterBand(band + 1).WriteArray(arr[:,:,band])
            src_mem.GetRasterBand(band + 1).SetNoDataValue(noData)

    # open target tif to extract spatial info
    ref_ds = gdal.Open(target_tif_path)
    gt_ref = ref_ds.GetGeoTransform()
    proj_ref = ref_ds.GetProjection()
    cols_ref = ref_ds.RasterXSize
    rows_ref = ref_ds.RasterYSize

    # warp it
    warped_ds = gdal.Warp('', src_mem,
              format='MEM',
              dstSRS=proj_ref,
              xRes=gt_ref[1],
              yRes=abs(gt_ref[5]),
              outputBounds=(
                  gt_ref[0],
                  gt_ref[3] + gt_ref[5] * rows_ref,
                  gt_ref[0] + gt_ref[1] * cols_ref,
                  gt_ref[3]
              ),
              resampleAlg=resamp)
    

    # export if path provided
    if output_path:
    
        if arrDim == 2:
            out_ds = gdal.GetDriverByName('GTiff').Create(output_path or '', cols_ref, rows_ref, 1, gdal.GDT_Float32)
            out_ds.SetGeoTransform(gt_ref)
            out_ds.SetProjection(proj_ref)
        else:
            out_ds = gdal.GetDriverByName('GTiff').Create(output_path or '', cols_ref, rows_ref, arr.shape[2], gdal.GDT_Float32)
            out_ds.SetGeoTransform(gt_ref)
            out_ds.SetProjection(proj_ref)

        # Return array
    if arrDim == 2:
        warped_array = warped_ds.GetRasterBand(1).ReadAsArray()
        if output_path:
            out_ds.GetRasterBand(1).WriteArray(warped_array)
            out_ds.GetRasterBand(1).SetNoDataValue(noData)
    else:
        warpL = []
        for band in range(arr.shape[2]):
            warpL.append(warped_ds.GetRasterBand(band + 1).ReadAsArray())
        warped_array = np.dstack(warpL)
        if output_path:
            for band in range(arr.shape[2]):
                out_ds.GetRasterBand(band + 1).SetNoDataValue(noData)
                out_ds.GetRasterBand(band + 1).WriteArray(warped_array[:,:,band])

    return warped_array


def get_ssrdsc_warped_and_corrected_at_doy(path_to_ssrdsc_grib, lst_acq_file, doy, slope_path,
                                           aspect_path, dem_path, lat_path, lon_path, outPath='MEM'):
    """will get the surface_solar_radiation_downward_clear_sky value (interpolated between 2 neighbouring values) at the time of LST acquisition
    per pixel for one LST composite (1 day). Furthermore, terrain correction and clear sky correction is done
    
    Returns a 2D numpy array

    Args:
        path_to_ssrdsc_grib (str): full path to ssrdsc file
        lst_acq_file (str): file to acquisition LST file (format: ...../filename_year_month.tif)
        doy (int): the day of lst_acq_file that should be processed --> lst_acq_file has one band per day
        slope_path (str): full path to slope (LST spat. resolution)
        aspect_path (str): full path to aspect (LST spat. resolution)
        dem_path (str): full path to dem (LST spat. resolution)
        lat_path (str): full path to lat (LST spat. resolution)
        lon_path (str): full path to lon (LST spat. resolution)
        output_path (str): path to filename where the raster stacks will be stored. If output_path == 'MEM', the warped raster will be returned in memory

    Returns:
        _type_: _description_
    """

    if outPath == 'MEM':
        era_ds = warp_ERA5_to_reference(path_to_ssrdsc_grib, lst_acq_file, outPath)
    else:
        warp_ERA5_to_reference(path_to_ssrdsc_grib, lst_acq_file, outPath)
        era_ds = gdal.Open(outPath)

        bandNumber = era_ds.RasterCount
        era_time = [pd.Timestamp(era_ds.GetRasterBand(i+1).GetDescription()) for i in range(bandNumber)]

        # open and load the acquisition raster from LST
        LST_acq_ds = gdal.Open(lst_acq_file)
        arr = LST_acq_ds.GetRasterBand(doy).ReadAsArray()
        arr = arr.astype(np.int64)
        arr_flat = arr.ravel()
        arr_ts = pd.to_datetime(arr_flat, unit='s')
        arr_up = pd.Series(arr_ts).dt.ceil('h')
        arr_up = arr_up.values.reshape(arr.shape)
        arr_min = pd.Series(arr_ts).dt.minute
        arr_min = arr_min.values.reshape(arr.shape)

        # get the relevant bands of era5 dataset (i.e. those bands, that reflect the upper and lower )
        bands = []
        for d1 in np.unique(arr_up):
            for count, e5 in enumerate(era_time):
                if d1 == e5:
                    bands.append(count)

        # load the era5 variable acquisition-wise into 2D numpy array 
        arrL = []
        for b in bands:
            # compare the time of LST composite with 
            mask = arr_up == era_time[b]
            # load the band that holds the observation just after LST acquisition
            after = era_ds.GetRasterBand(b).ReadAsArray()
            # load the band that holds the observation just prior to LST acquisition
            before = era_ds.GetRasterBand(b-1).ReadAsArray()
            # calculate the linearly interpolated values at the minute of acquisition
            vals = before - (before - after) * (np.array(arr_min, dtype=np.float16) / 60)
            arrL.append(vals * mask)
        block = np.dstack(arrL)

        block[block == 0] = np.nan
        ssrd = np.nanmax(block, axis = 2)
        ssrd_watt = ssrd / 3600

        # make a mask of the nan values, as the output of pvlib.irradiance.get_total_irradiance does skip them which hinders rebuild to 2D
        valid_mask = np.isfinite(ssrd_watt)
        arr_ts_masked = arr_ts[valid_mask.ravel()]

        # load slope, aspect, lat, lon, dem
        ds = gdal.Open(slope_path)
        slope = ds.GetRasterBand(1).ReadAsArray()
        slope_flat = slope.ravel()
        slope_flat_masked = slope_flat[valid_mask.ravel()]

        ds = gdal.Open(aspect_path)
        aspect = ds.GetRasterBand(1).ReadAsArray()
        aspect_flat = aspect.ravel()
        aspect_flat_masked = aspect_flat[valid_mask.ravel()]

        ds = gdal.Open(dem_path)
        dem = ds.GetRasterBand(1).ReadAsArray()
        dem_flat = dem.ravel()
        dem_flat_masked = dem_flat[valid_mask.ravel()]

        ds = gdal.Open(lat_path)
        lat = ds.GetRasterBand(1).ReadAsArray()
        lat_flat = lat.ravel()
        lat_flat_masked = lat_flat[valid_mask.ravel()]

        ds = gdal.Open(lon_path)
        lon = ds.GetRasterBand(1).ReadAsArray()
        lon_flat = lon.ravel()
        lon_flat_masked = lon_flat[valid_mask.ravel()]

        # get solar viewing conditions for acquisition time
        site = Location(lat_flat_masked, lon_flat_masked, altitude=dem_flat_masked, tz='UTC')
        solpos = site.get_solarposition(arr_ts_masked)

        # calculate extraterrestrial radiation (horizontal)
        dni_extra = pvlib.irradiance.get_extra_radiation(arr_ts_masked)

        # Compute clearness index
        ssrd_watt_flat = ssrd_watt.ravel()
        ssrd_watt_flat_masked = ssrd_watt_flat[valid_mask.ravel()]
        ghi = ssrd_watt_flat_masked
        cos_zenith = np.cos(np.radians(solpos['zenith'].values[0]))
        ghi_clear = dni_extra.values[0] * cos_zenith
        kt = ghi / ghi_clear

        # Decompose GHI to DNI and DHI using Erbs model
        dni, dhi = pvlib.irradiance.erbs(ghi, solpos['zenith'], arr_ts_masked)['dni'], \
                pvlib.irradiance.erbs(ghi, solpos['zenith'], arr_ts_masked)['dhi']

        # compute radiation on the tilted terrain

        irradiance_tilted = get_total_irradiance(
            surface_tilt=slope_flat_masked,
            surface_azimuth=aspect_flat_masked,
            dni=dni,
            ghi=ghi,
            dhi=dhi,
            solar_zenith=solpos['zenith'],
            solar_azimuth=solpos['azimuth']
        )

        # bring back to 2D
        poa_global_full = np.full(slope.shape, np.nan)
        poa_global_full[valid_mask] = irradiance_tilted['poa_global']

        return poa_global_full