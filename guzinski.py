from genericpath import exists
from helperToolz.helpsters import *
from helperToolz.dicts_and_lists import *
from helperToolz.mirmazloumi import *
import numpy as np
from osgeo import gdal, osr
from datetime import datetime, timezone
import re
import time
import pandas as pd
import pvlib
from pvlib import solarposition
from pvlib.irradiance import get_total_irradiance
from pvlib.location import Location
from scipy.interpolate import NearestNDInterpolator
from other_repos.pyDMS.pyDMS.pyDMS import *
from other_repos.pyTSEB.pyTSEB import meteo_utils
from other_repos.pyTSEB.pyTSEB import resistances
from other_repos.pyTSEB.pyTSEB import net_radiation
from other_repos.pyTSEB.pyTSEB import clumping_index 
from other_repos.pyTSEB.pyTSEB import TSEB
from other_repos.S2LP_FORCE.tools import SL2P, dictionariesSL2P
from other_repos.S2LP_FORCE.tools.read_sentinel2_force_image import *
from collections import defaultdict




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

    doy = [i+1 for i, day in enumerate(day_sequence) if day == compDate][0]

    # check for the special case, when compDate is YYYY0406 and dayrange == 4 as with this setting 1 April would not be processed
    # due to the FORCE processing with TSI at 9 day intervall that happened
    if (add_to_april == 1 and doy == 97) or (add_to_april == 0 and doy == 96):
        dayrange_low = dayrange + 1
        dayrange_up = dayrange
    else:
        dayrange_low = dayrange
        dayrange_up = dayrange

    # # check for the special case, when compDate is YYYY1027 and dayrange == 4 as with this setting November would be processed
    # if (add_to_april == 1 and doy >= 302) or (add_to_april == 0 and doy >= 301):
    #     dayrange_low = dayrange
    #     dayrange_up = (add_to_april + 304 - doy)
    # else:
    #     dayrange_low = dayrange
    #     dayrange_up = dayrange


    return {k: v for k, v in dicti.items() if doy - dayrange_low <= k <= doy + dayrange_up}


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


def temp_pressure_checker(list_of_era5_variables):
    """
    Ensure that 2m_temperature will be processed before surface pressure, as it is needed in the correction for the latter one

    Args:
        list_of_era5_variables (list): list of ERA5 variables

    Returns:
        print statement when check is finished
    """
    for idx, variable in enumerate(list_of_era5_variables):
        if '2m_temperature' in variable:
            temp_ind = idx
        elif 'surface_pressure'in variable:
            press_ind = idx
        else:
            pass
    
    if temp_ind < press_ind:
        pass
        # print('temperature will be processed before surface pressure - continue')
    else:
        cont = list_of_era5_variables[press_ind]
        list_of_era5_variables[press_ind] = list_of_era5_variables[temp_ind]
        list_of_era5_variables[temp_ind] = cont
        # print('2m_temperature and surface pressure swaped and now in right order - continue')


def warp_ERA5_to_reference(grib_path, reference_path, output_path='MEM', bandL='ALL', resampling='bilinear', NoData=False, 
                           sharp_DEM=None, sharp_geopot=None, sharp_rate=None, sharp_blendheight=None, sharp_temp=None,sharpener=None):
    """Warps a ERA .gib file to an existing raster in terms of resolution, projection, extent and aligns to raster.
    Optionally, the output can be DEM sharpened to a blending height (if all parameters are provided) before export

    Args:
        grib_path (str): complete path to the grib file
        reference_path (str): complete path to the raster to which the grib dataset will be warped to
        output_path (str): path to filename where the raster stacks will be stored. If output_path == 'MEM', the warped raster will be returned in memory
        resampling (str, optional): _description_. Defaults to 'bilinear'.
        bandL (list_of_int, optional): Defaults to 'ALL'. A subset of bands needs to be provided as a list of integer values, where the integer represents the
                                        actual band Number, not the band index of a numpy array!!!!!
        NoData (int, optional): if not provided, the function will try to retrieve the NoData value from the dataset
        single (bool): if set to True, only the first band of ERA will be exported (--> geopotential)

    """
    # sanity check
    # print(f"bandL = {bandL}, type = {type(bandL)}")

    if bandL != 'ALL':
        if type(bandL) != list:
            raise TypeError('if band numbers are provided, they must be in a list')
   
    # Open datasets
  
    ref_ds = checkPath(reference_path)
    
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

    # translated_ds = gdal.Translate(
    # '',                     # output to memory
    # grib_path,              # input file
    # format='MEM',
    # bandList=[1, 3]         # select specific bands (1-based index)
    # )

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
    # warped_ds = gdal.Warp('', translated_ds, options=warp_options)
    warped_ds = gdal.Warp('', grib_path, options=warp_options)

    # true for all options
    if bandL == 'ALL':
        bandCount = warped_ds.RasterCount
        bandList = [rc+1 for rc in range(bandCount)]
    else:
        bandCount = len(bandL)
        bandList = bandL

    if output_path != 'MEM':
        gdriver = gdal.GetDriverByName('GTiff')
        outPut = output_path
    else:
        gdriver = gdal.GetDriverByName('MEM')
        outPut = ''

    out_ds = gdriver.Create(outPut,
                            warped_ds.RasterXSize, warped_ds.RasterYSize,
                            bandCount, gdal.GDT_Float32)
    
    out_ds.SetGeoTransform(warped_ds.GetGeoTransform())
    out_ds.SetProjection(warped_ds.GetProjection())
    

    if sharpener:
        # load sharpener datasets 
        dem_ds = checkPath(sharp_DEM)
        dem = dem_ds.GetRasterBand(1).ReadAsArray()
        geopot_ds = checkPath(sharp_geopot)
        geopot = geopot_ds.GetRasterBand(1).ReadAsArray()
        
        if sharpener == 'adiabatic':
            if all([sharp_DEM, sharp_geopot, sharp_blendheight, sharp_rate]):
                print('Adiabatic Sharpener will be applied')

                for i in bandList:
                    band = warped_ds.GetRasterBand(i)
                    bandname = datetime.fromtimestamp(int(band.GetMetadata()['GRIB_VALID_TIME']), tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                    in_arr = band.ReadAsArray()
                    out_ds.GetRasterBand(i).WriteArray(applyAdiabaticDEMsharpener(era5_temp=in_arr,
                                                                                    dem=dem,
                                                                                    geopot=geopot,
                                                                                    rate=sharp_rate,
                                                                                    bheight=sharp_blendheight))
                    out_ds.GetRasterBand(i).SetNoDataValue(nodat)
                    out_ds.GetRasterBand(i).SetDescription(bandname)
                
                if output_path != 'MEM':
                    del out_ds
                else:
                    return out_ds

          
            elif any([sharp_DEM, sharp_geopot, sharp_blendheight, sharp_rate]):
                raise ValueError('Not all datasets for sharpening provided - Please provide all or none sharpener parameter')
            
            else:
                raise ValueError('Something weird happened')
 
        elif sharpener == 'barometric':
            if all(x is not None for x in [sharp_DEM, sharp_geopot, sharp_blendheight, sharp_temp]):
                print('Barometric Sharpener will be applied')

                for i in bandList:
                    band = warped_ds.GetRasterBand(i)
                    bandname = datetime.fromtimestamp(int(band.GetMetadata()['GRIB_VALID_TIME']), tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                    in_arr = band.ReadAsArray()
                    out_ds.GetRasterBand(i).WriteArray(applyBarometricDEMsharpener(era5_sp=in_arr,
                                                                                    dem=dem,
                                                                                    geopot=geopot,
                                                                                    bheight=sharp_blendheight,
                                                                                    era5_corrected_Temp=sharp_temp))
                    out_ds.GetRasterBand(i).SetNoDataValue(nodat)
                    out_ds.GetRasterBand(i).SetDescription(bandname)
                
                if output_path != 'MEM':
                    del out_ds
                else:
                    return out_ds
                    
            elif any(x is not None for x in [sharp_DEM, sharp_geopot, sharp_blendheight, sharp_temp]):
                raise ValueError('Not all datasets for sharpening provided - Please provide all or none sharpener parameter')
            
            else:
                raise ValueError('Something weird happened')
        
        else:
            raise ValueError('unknown sharpening method')
     
    else:
        for i in bandList:
            band = warped_ds.GetRasterBand(i)
            bandname = datetime.fromtimestamp(int(band.GetMetadata()['GRIB_VALID_TIME']), tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            out_ds.GetRasterBand(i).WriteArray(band.ReadAsArray())
            out_ds.GetRasterBand(i).SetNoDataValue(nodat)
            out_ds.GetRasterBand(i).SetDescription(bandname)
                
        if output_path != 'MEM':
            del out_ds
        else:
            return out_ds


def get_warped_ERA5_at_doy(path_to_era_grib, reference_path, lst_acq_file, doy, resampling='bilinear', outPath='MEM', bandL='ALL',
                           sharp_blendheight=False, sharp_DEM=False, sharp_geopot=False, sharp_rate=False, sharp_temp=False, sharpener=False, nodat=None):
    """will get the era5 value (interpolated between 2 neighbouring values) at the time of LST acquisition per pixel for one LST composite (1 day)
    Returns a 2D numpy array
    Args:
        path_to_era_grib (str): full path to grib file (monthly)
        reference_path (str): complete path to the raster to which the grib dataset will be warped to
        lst_acq_file (str): file to acquisition LST file (format: ...../filename_year_month.tif)
        doy (int): the day of lst_acq_file that should be processed --> lst_acq_file has one band per day
        resampling (str, optional): _description_. Defaults to 'bilinear'.
        output_path (str): path to filename where the raster stacks will be stored. If output_path == 'MEM', the warped raster will be returned in memory
        bandL (list_of_int, optional): Defaults to 'ALL'. A subset of bands needs to be provided as a list of integer values, where the integer represents the
                                        actual band Number, not the band index of a numpy array!!!!!
        sharp_blendheight (int): Defaults to False. If sharpening should be applied, all 4 parameters must be provided
        sharp_DEM (str): _description_. Path to DEM.
        sharp_geopot (str): Path to geopotential file
        sharp_rate (constant): adiabatic rate to be chosen from constants
        sharp_temp (numpy_array_float): corrected era5 air temperature
        sharpener (str): either adiabatic or barometric
        output_path (str): path to filename where the raster stacks will be stored. If output_path == 'MEM', the warped raster will be returned in memory
    """
    if outPath == 'MEM':
        era_ds = warp_ERA5_to_reference(grib_path=path_to_era_grib, reference_path=reference_path, resampling=resampling,
                                        output_path=outPath, bandL=bandL,
                                        sharp_blendheight=sharp_blendheight, sharp_DEM=sharp_DEM,
                                        sharp_geopot=sharp_geopot, sharp_rate=sharp_rate, sharp_temp=sharp_temp, sharpener=sharpener,
                                        NoData = nodat)
    else:
        warp_ERA5_to_reference(grib_path=path_to_era_grib, reference_path=reference_path, resampling=resampling,
                               output_path=outPath, bandL=bandL,
                               sharp_blendheight=sharp_blendheight, sharp_DEM=sharp_DEM,
                               sharp_geopot=sharp_geopot, sharp_rate=sharp_rate, sharp_temp=sharp_temp, sharpener=sharpener,
                               NoData = nodat)
        era_ds = gdal.Open(outPath)
    
    bandNumber = era_ds.RasterCount
    era_time = [pd.Timestamp(era_ds.GetRasterBand(i+1).GetDescription()) for i in range(bandNumber)]

    # open and load the acquisition raster from LST
    LST_acq_ds = checkPath(lst_acq_file)
    
    # here should be a loop if we go for more than one day
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
        vals_masked = np.where(mask, vals, np.nan)
        arrL.append(vals_masked)
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
    src_ds = checkPath(arr_tif_path)
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
    ref_ds = checkPath(target_tif_path)
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


def get_ssrdsc_warped_and_corrected_at_doy(path_to_ssrdsc_grib, reference_path, lst_acq_file, doy, slope_path,
                                           aspect_path, dem_path, lat_path, lon_path, outPath='MEM', bandL='ALL', nodat =None):
    """will get the surface_solar_radiation_downward_clear_sky value (interpolated between 2 neighbouring values) at the time of LST acquisition
    per pixel for one LST composite (1 day). Furthermore, terrain correction and clear sky correction is done
    
    Returns a 2D numpy array

    Args:
        path_to_ssrdsc_grib (str): full path to ssrdsc file
        reference_path (str): complete path to the raster to which the grib dataset will be warped to
        lst_acq_file (str): file to acquisition LST file (format: ...../filename_year_month.tif)
        doy (int): the day of lst_acq_file that should be processed --> lst_acq_file has one band per day
        slope_path (str): full path to slope (LST spat. resolution)
        aspect_path (str): full path to aspect (LST spat. resolution)
        dem_path (str): full path to dem (LST spat. resolution)
        lat_path (str): full path to lat (LST spat. resolution)
        lon_path (str): full path to lon (LST spat. resolution)
        output_path (str): path to filename where the raster stacks will be stored. If output_path == 'MEM', the warped raster will be returned in memory
        bandL (list_of_int, optional): Defaults to 'ALL'. A subset of bands needs to be provided as a list of integer values, where the integer represents the
                                        actual band Number, not the band index of a numpy array!!!!!

    Returns:
        poa_global_arr --> terrain corrected solar radiance
        zenith_arr --> solar zenith at LST acquisition times
        azimuth_arr --> solar azimuth at LST acquisition times
    """
    if outPath == 'MEM':
        era_ds = warp_ERA5_to_reference(grib_path=path_to_ssrdsc_grib, reference_path=reference_path, output_path=outPath, NoData=nodat)
    else:
        warp_ERA5_to_reference(grib_path=path_to_ssrdsc_grib, reference_path=reference_path, output_path=outPath, NoData=nodat)
        era_ds = gdal.Open(outPath)

    bandNumber = era_ds.RasterCount
    era_time = [pd.Timestamp(era_ds.GetRasterBand(i+1).GetDescription()) for i in range(bandNumber)]

    # open and load the acquisition raster from LST
    LST_acq_ds = checkPath(lst_acq_file)
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
        after = era_ds.GetRasterBand(b+1).ReadAsArray() # changed 2026-01-12 from b
        # load the band that holds the observation just prior to LST acquisition
        before = era_ds.GetRasterBand(b).ReadAsArray() # changed 2026-01-12 from b-1
        # calculate the linearly interpolated values at the minute of acquisition
        vals_interpolated = before - (before - after) * (np.array(arr_min, dtype=np.float16) / 60) # in J/m²
        # convert to W/m²
        vals_watt = vals_interpolated /3600
        arrL.append(vals_watt * mask)
    block = np.dstack(arrL)

    block[block == 0] = np.nan
    ssrd_watt = np.nanmax(block, axis = 2)

    # make a mask of the nan values, as the output of pvlib.irradiance.get_total_irradiance does skip them which hinders rebuild to 2D
    valid_mask = np.isfinite(ssrd_watt)
    arr_ts_masked = arr_ts[valid_mask.ravel()]

    # load slope, aspect, lat, lon, dem
    ds = checkPath(slope_path)
    slope = ds.GetRasterBand(1).ReadAsArray()
    slope_flat = slope.ravel()
    slope_flat_masked = slope_flat[valid_mask.ravel()]

    ds = checkPath(aspect_path)
    aspect = ds.GetRasterBand(1).ReadAsArray()
    aspect_flat = aspect.ravel()
    aspect_flat_masked = aspect_flat[valid_mask.ravel()]

    ds = checkPath(dem_path)
    dem = ds.GetRasterBand(1).ReadAsArray()
    dem_flat = dem.ravel()
    dem_flat_masked = dem_flat[valid_mask.ravel()]

    ds = checkPath(lat_path)
    lat = ds.GetRasterBand(1).ReadAsArray()
    lat_flat = lat.ravel()
    lat_flat_masked = lat_flat[valid_mask.ravel()]

    ds = checkPath(lon_path)
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
    poa_global_arr = np.full(slope.shape, np.nan)
    poa_global_arr[valid_mask] = irradiance_tilted['poa_global']


    zenith_arr = np.full(slope.shape, np.nan)
    zenith_arr[valid_mask] = solpos['zenith'].to_numpy()

    azimuth_arr = np.full(slope.shape, np.nan)
    azimuth_arr[valid_mask] = solpos['azimuth'].to_numpy()

    # calculate also the daily ssrd mean 
    meanL = []
    for count, e5 in enumerate(era_time):
        if e5.day == doy:
            ssrd_hour = era_ds.GetRasterBand(count+1).ReadAsArray() # changed 2026-01-12 to +1
            if np.sum(ssrd_hour) == 0:
                continue
            else:
                time_hour = np.full((len(lat_flat_masked),), e5)
            # get solar viewing conditions for acquisition time
            
            solpos = solarposition.get_solarposition(
                time=time_hour,       # vector of timestamps
                latitude=lat_flat_masked, # vector of latitudes (same length as time or broadcastable)
                longitude=lon_flat_masked,
                altitude=dem_flat_masked
            )

                        # Compute clearness index
            ghi = ssrd_hour / 3600
            ghi_masked = ghi[valid_mask]
            # Decompose GHI to DNI and DHI using Erbs model
            dni_dhi = pvlib.irradiance.erbs(ghi_masked.ravel(), solpos['zenith'], e5)
            dni, dhi = dni_dhi['dni'], dni_dhi['dhi']

            # compute radiation on the tilted terrain
            irradiance_tilted = get_total_irradiance(
            surface_tilt=slope_flat_masked,
            surface_azimuth=aspect_flat_masked,
            dni=dni,
            ghi=ghi_masked.ravel(),
            dhi=dhi,
            solar_zenith=solpos['zenith'],
            solar_azimuth=solpos['azimuth']
            )

            # put back into raster shape
            irr_2D = np.full(slope.shape, np.nan)
            irr_2D[valid_mask] = irradiance_tilted['poa_global']
            meanL.append(irr_2D)

    ssrd_mean = np.nansum(np.dstack(meanL), axis=2) / 24 # np.nanmean(np.dstack(meanL), axis = 2)
    ssrd_mean = ssrd_mean.reshape(slope.shape)

    return poa_global_arr, zenith_arr, azimuth_arr, ssrd_watt, ssrd_mean # *3600 to bring back to J/m²
    

def applyAdiabaticDEMsharpener(era5_temp, dem, geopot, rate, bheight):
    """takes a era5 value at blending height and converts to surface temperature using a dem and a lapse rate
    follows the formula from Mohammad
    Args:
        era5_temp (np_array_float): contains era5 temperature at 100 meter
        dem (np_array_float): contains DEM values
        geopot (np_array_float): contains geopotential values
        rate (float): the lapse rate, e.g. constant DRY_ADIABAT
        bheight (int): the assumed heigth at which era5 was recorded
    """
    return era5_temp - rate * ((dem + bheight) - (2 + (geopot/GRAVITY)))


def applyBarometricDEMsharpener(era5_sp, dem, geopot, bheight, era5_corrected_Temp):
    """takes a era5 value at blending height and converts to surface pressures using barrometric pressure formula
    (adapted from CHATGPT)
    Args:
        era5_sp (np_array_float): contains era5 surface pressure at 100 meter
        dem (np_array_float): contains DEM values
        geopot (np_array_float): contains geopotential values
        bheight (int): the assumed heigth at which era5 was recorded
        era5_corrected_Temp (np_array_float): contains the adiabitcally corrected era5 tepmerature at 2m (must be calculated before!!!)
    """

    # convert geopotential to ERA5 terrain height
    z_era5 = geopot / GRAVITY   # [m]
    z_era5_ref = 2 + z_era5     # 2 m above ERA5 terrain
    z_dem_target = dem + bheight  # blending height above DEM

    # calc elevation difference between DEM-based blending height and ERA5 2m reference
    delta_z = z_dem_target - z_era5_ref

    # apply barometric formula
    exponent = (GRAVITY * MOLAR_MASS_AIR) / (UNIVERSAL_GAS * STANDARD_ADIABAT)
    ratio = (era5_corrected_Temp - STANDARD_ADIABAT * delta_z) / era5_corrected_Temp

    # Avoid invalid values (e.g. negative or zero in ratio)
    # ratio = np.maximum(ratio, 1e-5)

    ratio[ratio <= 0] = np.nan
    
    return era5_sp * ratio ** exponent


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


def interpol_Time(arr):
    """ takes an array with time in unix seconds and interpolates via nearest neighbour

    Args:
        arr (np int 64): array with unix seconds and gaps filled with 0
    """
    # Coordinates of all indices
    x, y = np.indices(arr.shape)

    # Mask valid points
    valid_mask = (~np.isnan(arr)) & (arr > 0)

    # Build nearest-neighbor interpolator
    interp = NearestNDInterpolator(list(zip(x[valid_mask], y[valid_mask])), arr[valid_mask])

    # Copy the original array
    filled = arr.copy()

    # Find missing points (NaN or 0)
    missing_mask = np.isnan(arr) | (arr <= 0)

    # Interpolate only those missing points
    filled[missing_mask] = interp(x[missing_mask], y[missing_mask])

    # Cast back to int 
    filled = filled.astype(np.int64)

    return filled


def compute_incidence_angle(time_warp, lat, lon, dem, slope_rad, aspect_rad):
    # flatten
    time_flat = time_warp.ravel()
    lat_flat = lat.ravel()
    lon_flat = lon.ravel()
    dem_flat = dem.ravel()
    shape = time_warp.shape

    # convert to tz-aware datetimes (assume unix seconds UTC)
    times = pd.to_datetime(time_flat, unit='s', utc=True)

    # handle nodata times (e.g. <= 0) -> mask them
    valid_time_mask = ~np.isnan(time_flat) & (time_flat > 0)

    incidence_flat = np.full(time_flat.shape, np.nan, dtype=np.float32)

    if not np.any(valid_time_mask):
        return incidence_flat.reshape(shape)

    # speed: compute solar position grouped by unique timestamps that actually occur
    unique_times, inverse_idx = np.unique(times[valid_time_mask], return_inverse=True)
    # For each unique time, compute solar position for all valid pixels with that time.
    # Note: pvlib accepts vectorized latitude/longitude arrays of same length as lat/lon slice.
    # We'll iterate unique times (usually few) — cheaper than computing per-pixel if timestamps repeat.
    valid_indices = np.nonzero(valid_time_mask)[0]  # indices in flattened arrays

    for ut_i, ut in enumerate(unique_times):
        # indices (in flattened arrays) that correspond to this unique time
        sel = valid_indices[inverse_idx == ut_i]
        if sel.size == 0:
            continue

        sel_lat = lat_flat[sel]
        sel_lon = lon_flat[sel]
        sel_dem = dem_flat[sel]

        # compute solar position for these points at single time (pvlib will vectorize lat/lon)

        times_ut = pd.DatetimeIndex([ut] * len(sel_lat))
        sp = solarposition.get_solarposition(time=times_ut, latitude=sel_lat, longitude=sel_lon, altitude=sel_dem)

        zen = np.deg2rad(sp['zenith'].values)   # array shaped like sel
        az  = np.deg2rad(sp['azimuth'].values)

        # slope/aspect for these pixels
        s = slope_rad.ravel()[sel]
        asp = aspect_rad.ravel()[sel]

        # compute cos theta_i
        cos_theta = (np.cos(zen) * np.cos(s) +
                     np.sin(zen) * np.sin(s) * np.cos(az - asp))

        # clip numeric roundoff
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        # compute incidence angle (deg)
        with np.errstate(invalid='ignore'):
            incidence_deg = np.rad2deg(np.arccos(cos_theta))

        incidence_flat[sel] = incidence_deg

    return incidence_flat.reshape(shape)


def runSharpi(highResFilename, lowResFilename, lowResMaskFilename, cv, movWin, regrat, outputFilename, useDecisionTree = True):
    commonOpts = {"highResFiles":               [highResFilename],
                    "lowResFiles":              [lowResFilename],
                    "lowResQualityFiles":         [lowResMaskFilename],
                    "lowResGoodQualityFlags":     [1],
                    "cvHomogeneityThreshold":     cv,
                    "movingWindowSize":           movWin,
                    "disaggregatingTemperature":  True}
    dtOpts =     {"perLeafLinearRegression":    True,
                    "linearRegressionExtrapolationRatio": round(regrat, 2)}
    sknnOpts =   {'hidden_layer_sizes':         (10,),
                    'activation':                 'tanh'}
    nnOpts =     {"regressionType":             REG_sklearn_ann,
                    "regressorOpt":               sknnOpts}

    start_time = time.time()
    opts = commonOpts.copy()
    if useDecisionTree:
        opts.update(dtOpts)
        disaggregator = DecisionTreeSharpener(**opts)
    else:
        opts.update(nnOpts)
        disaggregator = NeuralNetworkSharpener(**opts)


    disaggregator.trainSharpener()

    downscaledFile = disaggregator.applySharpener(highResFilename, lowResFilename)

    residualImage, correctedImage = disaggregator.residualAnalysis(downscaledFile, lowResFilename,
                                                                    lowResMaskFilename,
                                                                    doCorrection=True)


    if correctedImage is not None:
        outImage = correctedImage
    else:
        outImage = downscaledFile
    # outData = utils.binomialSmoother(outData)
    outFile = utils.saveImg(outImage.GetRasterBand(1).ReadAsArray(),
                            outImage.GetGeoTransform(),
                            outImage.GetProjection(),
                            f"{os.path.split(outputFilename)[0]}/Values/{os.path.split(outputFilename)[1]}")
    residualFile = utils.saveImg(residualImage.GetRasterBand(1).ReadAsArray(),
                                residualImage.GetGeoTransform(),
                                residualImage.GetProjection(),
                                f"{os.path.split(outputFilename)[0]}/Residuals/{os.path.split(outputFilename)[1].split('.')[0]}_resid{os.path.splitext(outputFilename)[1]}")

    outFile = None
    residualFile = None
    downsaceldFile = None

    # print(time.time() - start_time, "seconds")


def runEvapi(year, month, day, comp, sharp, s2Mask, lstMask, tile, tempDir, path_to_temp,
             path_to_sharp, mvwin, cv, regrat, evap_outFolder, S2path, printInterim=False, bio=False):

    # storPath_c = f'{evap_outFolder}{comp}_{year}_{month}_{day}_{mvwin}_{cv}_{regrat}_{lstMask}_{s2Mask}_{sharp}_{tile}_ET_Canopy_calc.tif'
    # storPath_s = f'{evap_outFolder}{comp}_{year}_{month}_{day}_{mvwin}_{cv}_{regrat}_{lstMask}_{s2Mask}_{sharp}_{tile}_ET_Soil_calc.tif'

    storPath_c_f = f'{evap_outFolder}{comp}_{year}_{month}_{day}_{mvwin}_{cv}_{regrat}_{lstMask}_{s2Mask}_{sharp}_{tile}_ET_Canopy_func.tif'
    storPath_s_f = f'{evap_outFolder}{comp}_{year}_{month}_{day}_{mvwin}_{cv}_{regrat}_{lstMask}_{s2Mask}_{sharp}_{tile}_ET_Soil_func.tif'
    
    # path to era5 raw data
    era5_path = '/data/Aldhani/eoagritwin/et/Auxiliary/ERA5/grib/'
    ssrd_mean_path = '/data/Aldhani/eoagritwin/et/Auxiliary/ERA5/ssrd_mean_calc/'

    # the DEM, SLOPE, ASPECT, LAT, LON will be used to sharpen some of the era5 variables (the the resolution of the DEM)
    dem_path = '/data/Aldhani/eoagritwin/et/Auxiliary/DEM/reprojected/DEM_GER_FORCE_WARP.tif' # epsg 4326
    slope_path = '/data/Aldhani/eoagritwin/et/Auxiliary/DEM/reprojected/SLOPE_GER_FORCE_WARP.tif' # epsg 4326
    aspect_path = '/data/Aldhani/eoagritwin/et/Auxiliary/DEM/reprojected/ASPECT_GER_FORCE_WARP.tif' # epsg 4326
    lat_path = '/data/Aldhani/eoagritwin/et/Auxiliary/DEM/reprojected/LAT_GER_FORCE_WARP.tif' # epsg 4326
    lon_path = '/data/Aldhani/eoagritwin/et/Auxiliary/DEM/reprojected/LON_GER_FORCE_WARP.tif' # epsg 4326

    # the geopotential is needed for the sharpening as well
    geopot_path = '/data/Aldhani/eoagritwin/et/Auxiliary/ERA5/tiff/low_res/geopotential/geopotential_low_res.tif' # epsg 4326
    

    # path_base to sharpenend folder and S2_comp
    sharp_pathbase = f'{path_to_sharp}Values/'
    s2_pathbase = path_to_temp

    # the LST acquisition time should determine which sharpened LST files are associatedto be processed (as they are associated with it)
    LST_acq_file = f'/data/Aldhani/eoagritwin/et/Sentinel3/LST/LST_values/Acq_time/{year}/Daily_AcqTime_{comp}_{year}_{month}.tif' # epsg 4326

    # the VZA at the time of LST acquisition is need
    VZA_at_acq_file = f'/data/Aldhani/eoagritwin/et/Sentinel3/VZA/comp/{comp}/{year}/Daily_VZA_{comp}_{year}_{month}.tif' # epsg 4326

    # sharpened LST
    LST_file = f'{sharp_pathbase}{comp}_{year}_{month}_{day:02d}_{mvwin}_{cv}_{regrat}_{s2Mask}_{sharp}_{lstMask}_{tile}.tif' 
    # for NDVI calculation (estimating LAI and others) and warping to S2 resolution, we use the S2 composite used for sharpening
    # S2_file = [file for file in getFilelist(s2_pathbase, 'vrt', deep=False) if f'HIGHRES_{comp}_{year}_{month}_{day:02d}' in file][0]
    S2_file = S2Path# [file for file in getFilelist(s2_pathbase, 'vrt', deep=True) if 'S2' in file][0]

    # find era5 file that matches the month of LST observation
    valid_variables = sorted(list(dict.fromkeys(file.split('/')[-2] for file in getFilelist(era5_path, '.grib', deep=True) \
                                    if not any(var in file for var in ['geopotential', 'total_column_water_vapour']))))

    # get a list for those era5 files that match the year and month of the provided LST acquisition file
    era5_path_list = find_grib_file(getFilelist(era5_path, '.grib', deep=True), LST_acq_file)
    era5_path_list = [path for path in era5_path_list if any(variable in path for variable in valid_variables)] # era5 are epsg 4326 and still will be after warping to doy
    temp_pressure_checker(era5_path_list)

    # warp datasets needed for calculations to the spatial extent of the sharpened LST
    LST_acq_spatial_sub = warp_raster_to_reference(source_path=LST_acq_file, reference_path=S2_file, output_path='MEM', resampling='near')
    VZA_at_acq_file_sub = warp_raster_to_reference(source_path=VZA_at_acq_file, reference_path=S2_file, output_path='MEM', resampling='near')
    dem_sub = warp_raster_to_reference(source_path=dem_path, reference_path=S2_file, output_path='MEM', resampling='bilinear')
    slope_sub  = warp_raster_to_reference(source_path=slope_path, reference_path=S2_file, output_path='MEM', resampling='bilinear')
    aspect_sub = warp_raster_to_reference(source_path=aspect_path, reference_path=S2_file, output_path='MEM', resampling='bilinear')
    lat_sub = warp_raster_to_reference(source_path=lat_path, reference_path=S2_file, output_path='MEM', resampling='bilinear')
    lon_sub = warp_raster_to_reference(source_path=lon_path, reference_path=S2_file, output_path='MEM', resampling='bilinear')
    geopot_sub = warp_raster_to_reference(source_path=geopot_path, reference_path=S2_file, output_path='MEM', resampling='bilinear')

    # print if needed
    if printInterim:
        if path_to_temp.endswith('/'):
            temp_path_sub = path_safe(f"{path_to_temp}evap_interim/")
        else:
            temp_path_sub = path_safe(f"{path_to_temp}/evap_interim/")


        id_tag = f'{comp}_{year}_{month}_{day}_{mvwin}_{cv}_{regrat}_{lstMask}_{s2Mask}_{sharp}_{tile}'
        LST_acq_spatial_arr = LST_acq_spatial_sub.GetRasterBand(day).ReadAsArray()
        VZA_at_acq_file_arr = VZA_at_acq_file_sub.GetRasterBand(day).ReadAsArray()
        dem_arr= dem_sub.GetRasterBand(1).ReadAsArray()
        slope_arr = slope_sub.GetRasterBand(1).ReadAsArray()
        aspect_arr = aspect_sub.GetRasterBand(1).ReadAsArray()
        lat_arr = lat_sub.GetRasterBand(1).ReadAsArray()
        lon_arr = lon_sub.GetRasterBand(1).ReadAsArray()
        geopot_arr = geopot_sub.GetRasterBand(1).ReadAsArray()


        npTOdisk(LST_acq_spatial_arr, S2_file, f'{temp_path_sub}LST_acq_spatial_sub_{id_tag}.tif', bands = 1, d_type=gdal.GDT_Int64)
        npTOdisk(VZA_at_acq_file_arr, S2_file, f'{temp_path_sub}VZA_at_acq_file_sub_{id_tag}.tif', bands = 1)
        npTOdisk(dem_arr, S2_file, f'{temp_path_sub}dem_sub_{id_tag}.tif', bands = 1)
        npTOdisk(slope_arr, S2_file, f'{temp_path_sub}slope_sub_{id_tag}.tif', bands = 1)
        npTOdisk(aspect_arr, S2_file, f'{temp_path_sub}aspect_sub_{id_tag}.tif', bands = 1)
        npTOdisk(lat_arr, S2_file, f'{temp_path_sub}lat_sub_{id_tag}.tif', bands = 1)
        npTOdisk(lon_arr, S2_file, f'{temp_path_sub}lon_sub_{id_tag}.tif', bands = 1)
        npTOdisk(geopot_arr, S2_file, f'{temp_path_sub}geopot_sub_{id_tag}.tif', bands = 1)

    # load the era5 variable into cache at LST resolution and read-in the modelled times (one time step per band)
    for path in era5_path_list:
        # print(f'processing {path}')
        # check if DEM sharpener needs to be applied
        if '100m_u_component_of_wind' in path:
            # do the warping without sharpening
            try:
                wind100_u = get_warped_ERA5_at_doy(path_to_era_grib=path, reference_path=LST_acq_spatial_sub, lst_acq_file=LST_acq_spatial_sub, doy=day)
            except Exception as e:
                with open(f'{tempDir}ERROR_{comp}_{year}_{month}_{day}_{mvwin}_{cv}_{regrat}_{lstMask}_{s2Mask}_{sharp}_{tile}_ET.log', 'a') as f:
                    f.write(f'{e}')
                return

        elif '100m_v_component_of_wind' in path:
                # do the warping without sharpening
            try:
                wind100_v = get_warped_ERA5_at_doy(path_to_era_grib=path, reference_path=LST_acq_spatial_sub, lst_acq_file=LST_acq_spatial_sub, doy=day)
            except Exception as e:
                with open(f'{tempDir}ERROR_{comp}_{year}_{month}_{day}_{mvwin}_{cv}_{regrat}_{lstMask}_{s2Mask}_{sharp}_{tile}_ET.log', 'a') as f:
                    f.write(f'{e}')
                return
        # elif 'geopotential' in path:
        #     # do the warping without sharpening
        #     geopot = get_warped_ERA5_at_doy(path_to_era_grib=path, lst_acq_file=LST_acq_file, doy=day)

        elif 'downward' in path: # terrain correction included
            try:
                ssrd, szenith, sazimuth, ssrd_nc, ssrd_mean_func = get_ssrdsc_warped_and_corrected_at_doy(path_to_ssrdsc_grib=path, reference_path=LST_acq_spatial_sub, 
                                                                                lst_acq_file=LST_acq_spatial_sub, doy=day, 
                                                                                slope_path=slope_sub,
                                                                                aspect_path=aspect_sub,
                                                                                dem_path=dem_sub,
                                                                                lat_path=lat_sub,
                                                                                lon_path=lon_sub)
            except Exception as e:
                with open(f'{tempDir}ERROR_{comp}_{year}_{month}_{day}_{mvwin}_{cv}_{regrat}_{lstMask}_{s2Mask}_{sharp}_{tile}_ET.log', 'a') as f:
                    f.write(f'{e}')
                return
            
        elif '2m_temperature' in path: # DEM and adiabatic sharpening, following Guzinski 2021
            try:
                air_temp = get_warped_ERA5_at_doy(path_to_era_grib=path, reference_path=LST_acq_spatial_sub, 
                                                lst_acq_file=LST_acq_spatial_sub, doy=day,
                                                sharp_blendheight=100,
                                                sharp_DEM=dem_sub,
                                                sharp_geopot=geopot_sub,
                                                sharp_rate=STANDARD_ADIABAT,
                                                sharpener='adiabatic')
            except Exception as e:
                with open(f'{tempDir}ERROR_{comp}_{year}_{month}_{day}_{mvwin}_{cv}_{regrat}_{lstMask}_{s2Mask}_{sharp}_{tile}_ET.log', 'a') as f:
                    f.write(f'{e}')
                return
            
        elif '2m_dewpoint_temperature' in path: # DEM and adiabatic sharpening, following Guzinski 2021
            try:
                dew_temp = get_warped_ERA5_at_doy(path_to_era_grib=path, reference_path=LST_acq_spatial_sub, 
                                                lst_acq_file=LST_acq_spatial_sub, doy=day,
                                                sharp_blendheight=100,
                                                sharp_DEM=dem_sub,
                                                sharp_geopot=geopot_sub,
                                                sharp_rate=MOIST_ADIABAT,
                                                sharpener='adiabatic')
            except Exception as e:
                with open(f'{tempDir}ERROR_{comp}_{year}_{month}_{day}_{mvwin}_{cv}_{regrat}_{lstMask}_{s2Mask}_{sharp}_{tile}_ET.log', 'a') as f:
                    f.write(f'{e}')
                return
            
        else: 
            # do warping with DEM sharpening only
            # sanity check
            if not 'surface_pressure' in path:
                raise ValueError('There is and unattended ERA5 variable in the loop - CHECK!!!!')
            else:
                try:
                    sp = get_warped_ERA5_at_doy(path_to_era_grib=path, reference_path=LST_acq_spatial_sub, 
                                                lst_acq_file=LST_acq_spatial_sub, doy=day,
                                                sharp_DEM=dem_sub,
                                                sharp_blendheight=100,
                                                sharp_geopot=geopot_sub,
                                                sharp_temp=air_temp,
                                                sharpener='barometric') / 100
                except Exception as e:
                    with open(f'{tempDir}ERROR_{comp}_{year}_{month}_{day}_{mvwin}_{cv}_{regrat}_{lstMask}_{s2Mask}_{sharp}_{tile}_ET.log', 'a') as f:
                        f.write(f'{e}')
                    return
          
    wind_speed_20 = calc_wind_speed(wind100_u, wind100_v) # check wind_u
    
    ds = gdal.Open(f'{ssrd_mean_path}surface_solar_radiation_downward_clear_sky_{year}_{int(MONTH_TO_02D[month])}')
    ssrd_mean = ds.GetRasterBand(day).ReadAsArray() / 3600
    
    # ssrd_mean_calc_20 = warp_np_to_reference(ssrd_mean, f'{ssrd_mean_path}surface_solar_radiation_downward_clear_sky_{year}_{int(MONTH_TO_02D[month])}', LST_file) # check this too!!!!!
    ssrd_mean_func_20 = ssrd_mean_func
    ssrd_20 = ssrd
    air_temp_20 = air_temp
    dew_temp_20 = dew_temp
    sp_20 = sp
    szenith_20 = szenith
    sazimuth_20 = sazimuth

    # calculate windspeed

    # load vza
    vza_ds = VZA_at_acq_file_sub
    vza_20 = vza_ds.GetRasterBand(day).ReadAsArray()

    # load sharpened LST
    lst_ds = gdal.Open(LST_file)
    lst_20 =lst_ds.GetRasterBand(1).ReadAsArray()

    del wind100_u, wind100_v, ssrd, air_temp, dew_temp, sp, szenith, sazimuth, ssrd_mean, ssrd_nc, ssrd_mean_func

    if printInterim:
        # npTOdisk(ssrd_mean_calc_20, LST_file, f'{temp_path_sub}SSRD_mean_calc_{id_tag}.tif', bands = 1)
        npTOdisk(ssrd_mean_func_20, LST_file, f'{temp_path_sub}SSRD_mean_func_{id_tag}.tif', bands = 1)
        npTOdisk(ssrd_20, LST_file, f'{temp_path_sub}SSRD_{id_tag}.tif', bands = 1)
        npTOdisk(air_temp_20, LST_file, f'{temp_path_sub}TEMP_{id_tag}.tif', bands = 1)
        npTOdisk(dew_temp_20, LST_file, f'{temp_path_sub}DEW_{id_tag}.tif', bands = 1)
        npTOdisk(sp_20, LST_file, f'{temp_path_sub}SP_{id_tag}.tif', bands = 1)
        npTOdisk(szenith_20, LST_file, f'{temp_path_sub}ZEN_{id_tag}.tif', bands = 1)
        npTOdisk(sazimuth_20, LST_file, f'{temp_path_sub}AZI_{id_tag}.tif', bands = 1)
        npTOdisk(wind_speed_20, LST_file, f'{temp_path_sub}WSPEED_{id_tag}.tif', bands = 1)
        npTOdisk(lst_20, LST_file, f'{temp_path_sub}LST_{id_tag}.tif', bands = 1)
        npTOdisk(vza_20, LST_file, f'{temp_path_sub}VZA_{id_tag}.tif', bands = 1)

    condition = (air_temp_20 > 0) & (dew_temp_20 > 0)  & (sp_20 > 0) & (szenith_20 > 0) & (sazimuth_20 > 0) & (wind_speed_20 > 0) & (lst_20 > 0) & (vza_20 > 0)
    ssrd_20[~condition] = np.nan
    # ssrd_mean_calc_20[~condition] = np.nan
    ssrd_mean_func_20[~condition] = np.nan
    air_temp_20[~condition] = np.nan
    dew_temp_20[~condition] = np.nan
    sp_20[~condition] = np.nan
    szenith_20[~condition] = np.nan
    sazimuth_20[~condition] = np.nan
    wind_speed_20[~condition] = np.nan
    # lst_20 = np.ma.masked_where(~condition, lst_20)
    # vza_20 = np.ma.masked_where(~condition, vza_20)
    lst_20[~condition] = np.nan
    vza_20[~condition] = np.nan

    if printInterim:
        # npTOdisk(ssrd_mean_calc_20, LST_file, f'{temp_path_sub}SSRD_mean_calc_masked_{id_tag}.tif', bands = 1)
        npTOdisk(ssrd_mean_func_20, LST_file, f'{temp_path_sub}SSRD_mean_func_masked_{id_tag}.tif', bands = 1)
        npTOdisk(ssrd_20, LST_file, f'{temp_path_sub}SSRD_masked_{id_tag}.tif', bands = 1)
        npTOdisk(air_temp_20, LST_file, f'{temp_path_sub}TEMP_masked_{id_tag}.tif', bands = 1)
        npTOdisk(dew_temp_20, LST_file, f'{temp_path_sub}DEW_masked_{id_tag}.tif', bands = 1)
        npTOdisk(sp_20, LST_file, f'{temp_path_sub}SP_masked_{id_tag}.tif', bands = 1)
        npTOdisk(szenith_20, LST_file, f'{temp_path_sub}ZEN_masked_{id_tag}.tif', bands = 1)
        npTOdisk(sazimuth_20, LST_file, f'{temp_path_sub}AZI_masked_{id_tag}.tif', bands = 1)
        npTOdisk(wind_speed_20, LST_file, f'{temp_path_sub}WSPEED_masked_{id_tag}.tif', bands = 1)
        npTOdisk(lst_20, LST_file, f'{temp_path_sub}LST_masked_{id_tag}.tif', bands = 1)
        npTOdisk(vza_20, LST_file, f'{temp_path_sub}VZA_masked_{id_tag}.tif', bands = 1)


    # calculate the NDVI from the S2 composite (following formula from force --> bandnames: (NIR - RED) / (NIR + RED))
    S2_ds = gdal.Open(S2_file)
    for idx, bname in enumerate(getBandNames(S2_file)):
        if bname == 'RED':
            red = S2_ds.GetRasterBand(1 + idx).ReadAsArray()
        elif bname == 'BNR':
            nir = S2_ds.GetRasterBand(1 + idx).ReadAsArray()
        else:
            continue

    # ndvi_20 = (nir - red) / (nir + red)
    # ndvi_20_ma = np.where(ndvi_20 < 0, np.nan, ndvi_20)
    # ndvi_20_ma = np.ma.masked_invalid(ndvi_20)
    # ndvi_20_ma = np.ma.masked_where(ndvi_20_ma < 0, ndvi_20_ma)

    albedo, ccc, cwc, lai, fapar, fcover = read_biophys(bio)
    theta = np.deg2rad(szenith_20)
    theta_safe = np.clip(theta, 0.05, 1.0)

    lai_min = 0.1
    valid = lai > lai_min

    Cab = np.full_like(lai, np.nan, dtype=float)
    Cw  = np.full_like(lai, np.nan, dtype=float)

    Cab[valid] = ccc[valid] / lai[valid]
    Cw[valid]  = cwc[valid] / lai[valid]

    fg, FIPAR, PAI = compute_fg_fipar_pai(
        LAI=lai,
        FAPAR=fapar,
        theta=theta_safe)

    # LAI_np = 0.57*np.exp(2.33*ndvi_20)
    # LAI_pos = np.where(LAI_np < 0, np.nan, LAI_np)
    LAI_np = lai
    LAI_pos = np.where(LAI_np < 0, np.nan, LAI_np)
    # estimate canopy height from estimated LAI
    hc = hc_from_lai(LAI_pos, hc_max = 1.2, lai_max = np.nanmax(LAI_np), hc_min=0)

    # estimate long wave irradiance
    ea = meteo_utils.calc_vapor_pressure(T_K=dew_temp_20)
    L_dn = calc_longwave_irradiance(ea = ea, t_a_k = air_temp_20, p = sp_20, z_T = 100, h_C = hc) # ## does that make sense with the 100m!!!!!!!!!!!!!!!!!!!
    d_0_0 = resistances.calc_d_0(h_C=hc)
    z_0 = resistances.calc_z_0M(h_C=hc)


    # calculate shortwave radiation of soil and canopy

    # difvis, difnir, fvis, fnir = net_radiation.calc_difuse_ratio(S_dn = ssrd_20, sza = np.nanmean(szenith_20))
    ssrd_1d = ssrd_20.ravel()
    sza_1d  = szenith_20.ravel()
    dem_arr= dem_sub.GetRasterBand(1).ReadAsArray()
    press_1d = 1013.25 * np.exp(-dem_arr.ravel() / 8434.5)
    difvis, difnir, fvis, fnir = net_radiation.calc_difuse_ratio(S_dn=ssrd_1d,sza=sza_1d,press=press_1d)

    difvis = difvis.reshape(ssrd_20.shape)
    difnir = difnir.reshape(ssrd_20.shape)
    fvis   = fvis.reshape(ssrd_20.shape)
    fnir   = fnir.reshape(ssrd_20.shape)


    skyl = difvis * fvis + difnir * fnir
    S_dn_dir = ssrd_20 * (1.0 - skyl)
    S_dn_dif = ssrd_20 * skyl

    # Leaf spectral properties:{rho_vis_C: visible reflectance, tau_vis_C: visible transmittance, rho_nir_C: NIR reflectance, tau_nir_C: NIR transmittance}
    Cab = ccc / np.maximum(lai, 1e-6)
    Cw  = cwc / np.maximum(lai, 1e-6)
    rho_vis_C, tau_vis_C = leaf_optics_vis(Cab)
    rho_nir_C, tau_nir_C = leaf_optics_nir(Cw)
    # rho_vis_C=np.full(LAI_pos.shape, 0.05, np.float32)
    # tau_vis_C=np.full(LAI_pos.shape, 0.08, np.float32)
    # rho_nir_C=np.full(LAI_pos.shape, 0.32, np.float32)
    # tau_nir_C=np.full(LAI_pos.shape, 0.33, np.float32) 

    # Soil spectral properties:{rho_vis_S: visible reflectance, rho_nir_S: NIR reflectance}
    rho_vis_S=np.full(LAI_pos.shape, 0.07, np.float32)
    rho_nir_S=np.full(LAI_pos.shape, 0.25, np.float32)

    # F = local LAI
    F = LAI_pos / 1
    # calculate clumping index
    f_c = np.clip(fcover, 0.05, 0.99)
    Omega0 = clumping_index.calc_omega0_Kustas(LAI = LAI_np, f_C = f_c, x_LAD=1)
    Omega = clumping_index.calc_omega_Kustas(Omega0, szenith_20)
    LAI_eff = F * Omega

    Sn_C, Sn_S = net_radiation.calc_Sn_Campbell(lai = LAI_pos, sza = szenith_20, S_dn_dir = S_dn_dir, S_dn_dif = S_dn_dif, fvis = fvis,
                                        fnir = fnir, rho_leaf_vis = rho_vis_C, tau_leaf_vis = tau_vis_C, rho_leaf_nir = rho_nir_C, 
                                        tau_leaf_nir = tau_nir_C, rsoilv = rho_vis_S, rsoiln = rho_nir_S, x_LAD=1, LAI_eff=LAI_eff)

    # calculate other roughness stuff

    landC = np.full(LAI_pos.shape, 11, dtype=np.int16)
    w_C = np.ones_like(LAI_pos, dtype=np.float32)
    z_0M, d = resistances.calc_roughness(LAI=LAI_pos, h_C=hc, w_C=w_C, landcover=landC, f_c=f_c)
    # fg = calc_fg_gutman(ndvi = ndvi_20_ma, ndvi_min = np.nanmin(ndvi_20), ndvi_max = np.nanmax(ndvi_20))

    if printInterim:
        # npTOdisk(ndvi_20, LST_file, f'{temp_path_sub}ndvi_{id_tag}.tif')
        # npTOdisk(ndvi_20_ma, LST_file, f'{temp_path_sub}ndvi_pos_{id_tag}.tif')
        npTOdisk(LAI_np, LST_file, f'{temp_path_sub}LAI_{id_tag}.tif')
        npTOdisk(LAI_pos, LST_file, f'{temp_path_sub}LAI_pos_{id_tag}.tif')
        npTOdisk(hc, LST_file, f'{temp_path_sub}canopy_height_{id_tag}.tif')
        npTOdisk(ea, LST_file, f'{temp_path_sub}vapor_pressure_{id_tag}.tif')
        npTOdisk(L_dn, LST_file, f'{temp_path_sub}longwave_radiation_{id_tag}.tif')
        npTOdisk(d_0_0, LST_file, f'{temp_path_sub}resistanceD_{id_tag}.tif')
        npTOdisk(z_0, LST_file, f'{temp_path_sub}resistanceZ_{id_tag}.tif')

        npTOdisk(Omega0, LST_file, f'{temp_path_sub}Omega0_{id_tag}.tif')
        npTOdisk(Omega, LST_file, f'{temp_path_sub}Omega_{id_tag}.tif')
        npTOdisk(Sn_C, LST_file, f'{temp_path_sub}Sn_C_{id_tag}.tif')
        npTOdisk(Sn_S, LST_file, f'{temp_path_sub}Sn_S_{id_tag}.tif')
        npTOdisk(fg, LST_file, f'{temp_path_sub}fg_{id_tag}.tif')

    emis_C = 0.98
    emis_S = 0.95
    h_C = hc 
    z_u = 100
    z_T = 100

    output = TSEB.TSEB_PT(lst_20, vza_20, air_temp_20, wind_speed_20, ea, sp_20, Sn_C, Sn_S, L_dn, LAI_pos, h_C, emis_C, emis_S, 
    z_0M, d, z_u, z_T, resistance_form=None, calcG_params=None, const_L=None, f_g=fg,
    kB=0.0, massman_profile=None, verbose=True)

    # for stori, ssrd_ras in zip([[storPath_c, storPath_s],[storPath_c_f, storPath_s_f]], [ssrd_mean_calc_20, ssrd_mean_func_20]):    
        # le_c = output[6]/ssrd_20
        # heat_latent_scaled_c = ssrd_ras * le_c
        # et_daily_c = TSEB.met.flux_2_evaporation(heat_latent_scaled_c, t_k=air_temp_20, time_domain=24)

        # le_s = output[8]/ssrd_20
        # heat_latent_scaled_s = ssrd_ras * le_s
        # et_daily_s = TSEB.met.flux_2_evaporation(heat_latent_scaled_s, t_k=air_temp_20, time_domain=24)

        # npTOdisk(et_daily_c, LST_file, stori[0])
        # npTOdisk(et_daily_s, LST_file, stori[1])
  
    le_c = output[6]/ssrd_20
    heat_latent_scaled_c = ssrd_mean_func_20 * le_c
    et_daily_c = TSEB.met.flux_2_evaporation(heat_latent_scaled_c, t_k=air_temp_20, time_domain=24)

    le_s = output[8]/ssrd_20
    heat_latent_scaled_s = ssrd_mean_func_20 * le_s
    et_daily_s = TSEB.met.flux_2_evaporation(heat_latent_scaled_s, t_k=air_temp_20, time_domain=24)

    npTOdisk(et_daily_c, LST_file, storPath_c_f)
    npTOdisk(et_daily_s, LST_file, storPath_s_f)


def Sharp_Evap(tile_to_process, storFolder, path_to_slope, path_to_aspect, path_to_agro, path_to_force,
               path_to_inci, path_to_lst, time_start, time_end, compList, predList, S2mask, printEvapInter=False,
               path_to_dem=False, path_to_lat=False, path_to_lon=False, path_to_acq=False, path_to_vaa=False, path_to_vza=False, 
               para_mode='perTILE', compDate=False):

    temp_dump_fold = path_safe(f"{storFolder}temp/{tile_to_process.replace('_', '')}/")
    sharp_outFolder = path_safe(f"{storFolder}sharpened/{tile_to_process.replace('_', '')}/")
    trash_path = path_safe(f"{storFolder}trash/")
    evap_outFolder = path_safe(f"{storFolder}evap/{tile_to_process.replace('_', '')}/")
    
    year = time_start[:4]

    # ############## make vrts for slope, aspect and agromask
    slopes = [file for file in getFilelist(path_to_slope, '.tif') if tile_to_process in file] # if any tile name is in file
    # aspect-tiles
    aspects = [file for file in getFilelist(path_to_aspect, '.tif') if tile_to_process in file] # if any tile name is in file
    # thuenen-tiles
    thuenen = [file for file in getFilelist(path_to_agro, '.tif') if tile_to_process in file] # if any tile name is in file
    
    # get those tiles (and composite if more than one tile is provided)
    slope_path = f'{temp_dump_fold}SLOPE.vrt'
    gdal.BuildVRT(slope_path, slopes)

    aspect_path = f'{temp_dump_fold}ASPECT.vrt'
    gdal.BuildVRT(aspect_path, aspects)

    thuenen_path = f'{temp_dump_fold}THUENEN.vrt'
    gdal.BuildVRT(thuenen_path, thuenen)

        # needed to estimate biophysical parameter
    if path_to_dem:
        # dem-tiles
        dems = [file for file in getFilelist(path_to_dem, '.tif') if tile_to_process in file] # if any tile name is in file
        # lat-tiles
        lats = [file for file in getFilelist(path_to_lat, '.tif') if tile_to_process in file] # if any tile name is in file
        # lon-tiles
        lons = [file for file in getFilelist(path_to_lon, '.tif') if tile_to_process in file] # if any tile name is in file

        dem_path = f'{temp_dump_fold}DEM.vrt'
        gdal.BuildVRT(dem_path, dems)

        lat_path = f'{temp_dump_fold}LAT.vrt'
        gdal.BuildVRT(lat_path, lats)
                    
        lon_path = f'{temp_dump_fold}LON.vrt'
        gdal.BuildVRT(lon_path, lons)


    # make the mask ready for S2 masking
    th_ds = gdal.Open(thuenen_path)
    th_arr = th_ds.GetRasterBand(1).ReadAsArray()
    mask = np.where(th_arr == -9999, 0, 1)

    colors = ['BLU', 'GRN', 'RED', 'BNR', 'NIR', 'RE1', 'RE2', 'RE3',  'SW1', 'SW2']

    # ################ load force and vrt
    path_to_S2_tiles = f'{path_to_force}/{year}/'
    
    if para_mode == 'perTILE':
        # get a list with all available tiles
        files = getFilelist(f'{path_to_S2_tiles}', '.tif', deep=True) 
        files = [file for file in files if tile_to_process in file]
        date_list = check_forceTSI_compositionDates(files)



        # check, if dates for this tile have already been processed, e.g. in a prior process that got terminated unexpectedly
        csv_path = f"{evap_outFolder}compdates.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            dateList_for_csv = df['compdates'].tolist()
        else:
            dateList_for_csv = []

        # this should be the masterloop within the sharpend, evaping and deleting of all files but the above created vrts takes place
        for date in date_list:

            if int(time_start) <= int(date) <= int(time_end): # neede to be adapted in a manner that incorporates the +/-4 otherwise time_start 20190401 
                # would find date 20190406 as int(time_start) <= int(date) <= int(time_end)..

                # take care of the csv-list that contains the dates of date_list. if the current date is already on an existing list, it probably has already
                # been processed. However, a prior process might have stopped while running this date. Therefore, we check if the date is the last entry

                if int(date) in dateList_for_csv and int(date) != dateList_for_csv[-1]:
                    print(f"{date} already processed")
                    continue
                elif int(date) in dateList_for_csv and int(date) == dateList_for_csv[-1]:
                    print(f"{date} already processed, but checking if all is there")
                    # check if date is the last in datelist, and if not, check if the S2_vrt of the following date is already in the folder (if yes, this date
                    # had been processed already sucessfully)
                    if int(date) != date_list[-1]:
                        if os.path.exists(f"{temp_dump_fold}S2_{date_list[date_list.index(date) + 1]}.vrt"):
                            continue
                else:
                    dateList_for_csv.append(date)
                    print(f"working on {date}")

                lowRes_files = []
                highRes_files = []
                highRes_names = []
                
                # get S2 bands
                tilesS2 = [file for file in getFilelist(path_to_S2_tiles, '.tif', deep=True) if tile_to_process in file and f'{date}.tif' in file]
                tilesS2 = [t2 for col in colors for t2 in tilesS2 if col in t2]
                S2_path = f'{temp_dump_fold}S2_{date}.vrt'
                
                # build vrt
                vrt = gdal.BuildVRT(S2_path, tilesS2, separate=True)
                vrt = None
                vrt = gdal.Open(S2_path, gdal.GA_Update)  # VRT must be writable
                for idz, bname in enumerate(colors): 
                    band = vrt.GetRasterBand(1+idz)
                    band.SetDescription(bname)
                vrt = None

                # determine LST and incidence files associated with respective S2 composite
                band_dict = transform_compositeDate_into_LSTbands(date, 4)

                # stat used for compositing
                for comp_stat in compList: #  
                    path_to_incident = f'{path_to_inci}{comp_stat}/{year}/'
                    path_to_LST = f'{path_to_lst}{comp_stat}/{year}/'

                    # get all LST bands that can be sharped with the S2 composite at this date (and sun angle incidence files as well, as they are dependent on that date
                    for k, v in band_dict.items(): # basically, this is a loop over the days 
                        month = v['month']
                        band = int(v['band'])
                        v_path = f'{path_to_LST}Daily_LST_{comp_stat}_{year}_{month}.tif'
                        ds = gdal.Open(v_path, 0)

                        if path_to_dem:
                            # calculate biophysical helper and parameter
                            try:
                                calc_biophys_helper(acq_path=path_to_acq, dem_path=dem_path, lat_path=lat_path, lon_path=lon_path,
                                                    vaa_path=path_to_vaa, vza_path=path_to_vza, year=year, month=month,
                                                    doy=band, outPath=temp_dump_fold, comp_stat=comp_stat)
                            except Exception as e:
                                print(f'broken because: {e} path_dem={path_to_dem} comp={comp_stat} month={month} day={band} tile={tile_to_process}')
                                break

                            get_biophysical_parameter(path_S2=tilesS2, path_Sun=f"{temp_dump_fold}sun/", year=year, month=month, doy=band,
                                                    outPath=temp_dump_fold)
                            
                        # export the LST for that day
                        LST_arr = ds.GetRasterBand(band).ReadAsArray() # store as single Tiff in temp
                        daily_lst_path = f'{temp_dump_fold}Daily_LST_{comp_stat}_{year}_{month}_{band:02d}.tif'
                        makeTif_np_to_matching_tif(LST_arr, v_path, daily_lst_path)
                    
                        # store the paths for selecting incidence for corresponding LST
                        incid_date = f'{year}_{month}_{band:02d}.tif'

                        # incidence-tiles
                        # incids = [file for file in getFilelist(path_to_incident, '.tif', deep=True) if tile_to_process in file] 
                        # incid_path = incids[0]
                        incid_path = [file for file in getFilelist(path_to_incident, '.tif', deep=True) if f'{tile_to_process}_{incid_date}' in file][0]

                        # create highRes file through exapnding the vrt of S2
                        highRes_path = f"{temp_dump_fold}HIGHRES_{comp_stat}_{incid_date.split('.')[0]}.vrt"
                        gdal.BuildVRT(highRes_path, [S2_path, slope_path, aspect_path, incid_path], separate=True)

                        for predi in predList: # different predictor combinations for the sharpening
                            if predi == 'allpred':
                                maskVRT_water(highRes_path, colorlist=colors)
                            else:
                                maskVRT_water_and_drop_aux(highRes_path, colorlist=colors)

                            if S2mask == 1:
                                highRes_files.append(f"{highRes_path.split('.')[0]}_watermask.tif")
                                highRes_names.append(f'S2notMasked_{predi}')
                                lowRes_files.append(daily_lst_path)

                            elif S2mask == 2:
                                maskVRT(f"{highRes_path.split('.')[0]}_watermask.tif", mask, suffix=f"_S2_agromask_{predi}")
                                os.remove(f"{highRes_path.split('.')[0]}_watermask.tif")
                                highRes_files.append(f"{highRes_path.split('.')[0]}_watermask_S2_agromask_{predi}.tif")
                                lowRes_files.append(daily_lst_path)
                                highRes_names.append(f"S2Masked_{predi}")

                            elif S2mask == 3:
                                highRes_files.append(f"{highRes_path.split('.')[0]}_watermask.tif")
                                highRes_names.append(f"S2notMasked_{predi}")
                                lowRes_files.append(daily_lst_path)
                                maskVRT(f"{highRes_path.split('.')[0]}_watermask.tif", mask, suffix=f"_S2_agromask_{predi}")
                                highRes_files.append(f"{highRes_path.split('.')[0]}_watermask_S2_agromask_{predi}.tif")
                                lowRes_files.append(daily_lst_path)
                                highRes_names.append(f"S2Masked_{predi}")
                
                # sharp all files for that day and all compstats
                sharpList = []

                for idx, highResFilename in enumerate(highRes_files):
                        lowResFilename = lowRes_files[idx]
                        # f1 = f'{sharp_outFolder}{'/'.join(highResFilename.split('.')[0].split('_')[2:5])}/'
                        for maskname, mask_lowRes in zip(['withoutLSTmask'], ['']): # 'withLSTmask'  lowmask_bin_path
                        # for maskname, mask_lowRes in zip(['withoutLSTmask', 'withLSTmask'], ['', lowmask_bin_path]):
                            lowResMaskFilename = mask_lowRes
                            # f2 = f'{f1}{maskname}/'
                            for movWin in [15]:
                                for cv in [0]:
                                    for regrat in [0.25]:
                                        kombi = f'mvwin{movWin}_cv{cv}_regrat{int(regrat*100):02d}_{highRes_names[idx]}_{maskname}'
                                        # f3 = f'{f2}{highRes_names[idx]}/'
                                        # os.makedirs(f'{f3}Residuals/', exist_ok=True)
                                        # os.makedirs(f'{f3}Values/', exist_ok=True)

                                        os.makedirs(f'{sharp_outFolder}Residuals/', exist_ok=True)
                                        os.makedirs(f'{sharp_outFolder}Values/', exist_ok=True)
                                        
                                        # sharpened_file = f'{f3}{'_'.join(highResFilename.split('.')[0].split('_')[2:6])}_{kombi}_{tile_to_process}.tif'
                                        sharpened_file = f"{sharp_outFolder}{'_'.join(highResFilename.split('.')[0].split('_')[1:5])}_{kombi}_{tile_to_process}.tif"
                                        
                                        # check if evap file for sharpened_file already exists (might happen when we restart a process that terminated unexpectedly
                                        # before and now we go through the last date from datelist that had been processed)
                                        file_name = f"{evap_outFolder}{sharpened_file.split('/')[-1].split('_')[0]}_{sharpened_file.split('/')[-1].split('_')[1]}\
                                            _{sharpened_file.split('/')[-1].split('_')[2]}_{int(sharpened_file.split('/')[-1].split('_')[3])}_{movWin}_{cv}_{regrat}_\
                                                {sharpened_file.split('/')[-1].split('_')[9]}_{sharpened_file.split('/')[-1].split('_')[7]}_\
                                                    {sharpened_file.split('/')[-1].split('_')[8]}_{tile_to_process}"
                                        

                                        if (os.path.exists(f"{file_name}_ET_Canopy_func.tif") and os.path.exists(f"{file_name}_ET_Soil_func.tif")):
                                            print(f"{file_name} already created")
                                            continue

                                        runSharpi(highResFilename, lowResFilename, lowResMaskFilename, cv, movWin, regrat, sharpened_file, useDecisionTree = True)
                                        
                                        # run the biophysical parameter calculation here
                                        
                                        # should run the function and return a list with the paths for all the created files
                                        # this list will then be passed on to runEvapi
                                        sharpList.append(sharpened_file)
                
                # calc evapo for all compstats at that day
                for sharped in sharpList:
                    comp = sharped.split('/')[-1].split('_')[0]
                    year = sharped.split('/')[-1].split('_')[1]
                    month = sharped.split('/')[-1].split('_')[2]
                    day = int(sharped.split('/')[-1].split('_')[3])
                    mvwin = sharped.split('/')[-1].split('_')[4]
                    cv = sharped.split('/')[-1].split('_')[5]
                    regrat = sharped.split('/')[-1].split('_')[6]
                    sharp = sharped.split('/')[-1].split('_')[8]
                    s2Mask = sharped.split('/')[-1].split('_')[7]
                    lstMask = sharped.split('/')[-1].split('_')[9]
                    tile = '_'.join(sharped.split('/')[-1].split('.')[0].split('_')[-2:])

                    if path_to_dem:
                        bio_pars = [file for file in getFilelist(f'{temp_dump_fold}bio/', '.tif') if f'{year}_{month}_{day}.tif' in file]
                    else:
                        bio_pars = False

                    

                    runEvapi(year=year, month=month, day=day, comp=comp, sharp=sharp, s2Mask=s2Mask, lstMask=lstMask, tile=tile,
                            tempDir=trash_path, path_to_temp=temp_dump_fold, path_to_sharp=sharp_outFolder, mvwin=mvwin, cv=cv,
                            regrat=regrat, evap_outFolder=evap_outFolder, printInterim=printEvapInter, bio=bio_pars)
                
                
                # at the end of the date loop --> clean up temp folder and sharp
                temp_files_vrt = [file for file in getFilelist(temp_dump_fold, '.vrt')]
                temp_files_tif = [file for file in getFilelist(temp_dump_fold, '.tif')]

                if path_to_dem:
                    [temp_files_vrt.remove(path) for path in [slope_path, aspect_path, thuenen_path, dem_path, lat_path, lon_path]]
                    bio_files = [file for file in getFilelist(f'{temp_dump_fold}bio/', '.tif')]
                    sun_files = [file for file in getFilelist(f'{temp_dump_fold}sun/', '.tif')]
                    [os.remove(file) for file in bio_files]
                    [os.remove(file) for file in sun_files]

                else:    
                    [temp_files_vrt.remove(path) for path in [slope_path, aspect_path, thuenen_path]]
                sharp_files = [file for file in getFilelist(sharp_outFolder, '.tif', deep=True)]
                [os.remove(file) for file in temp_files_vrt]
                [os.remove(file) for file in temp_files_tif]
                [os.remove(file) for file in sharp_files]

            
                # export the processed dates
                df = pd.DataFrame({
                    'compdates': dateList_for_csv,
                    })
                df.to_csv(csv_path, index=False)

    elif para_mode == 'perDATE':
        if compDate == False:
            print('NO SINGLE DATE PROVIDED - THIS PROCESS WILL CRASH SOON :(')
        
        compdate_path = f"{evap_outFolder}compdates_{compDate}.csv"

        if int(time_start) <= int(compDate) <= int(time_end): # neede to be adapted in a manner that incorporates the +/-4 otherwise time_start 20190401 
            # would find date 20190406 as int(time_start) <= int(date) <= int(time_end)..

            lowRes_files = []
            highRes_files = []
            highRes_names = []
            
            # get S2 bands
            tilesS2 = [file for file in getFilelist(path_to_S2_tiles, '.tif', deep=True) if tile_to_process in file and f'{compDate}.tif' in file]
            tilesS2 = [t2 for col in colors for t2 in tilesS2 if col in t2]
            S2_path = f'{temp_dump_fold}S2_{compDate}.vrt'
            
            # build vrt
            vrt = gdal.BuildVRT(S2_path, tilesS2, separate=True)
            vrt = None
            vrt = gdal.Open(S2_path, gdal.GA_Update)  # VRT must be writable
            for idz, bname in enumerate(colors): 
                band = vrt.GetRasterBand(1+idz)
                band.SetDescription(bname)
            vrt = None

            # determine LST and incidence files associated with respective S2 composite
            band_dict = transform_compositeDate_into_LSTbands(compDate, 4)

            # stat used for compositing
            for comp_stat in compList: #  
                path_to_incident = f'{path_to_inci}{comp_stat}/{year}/'
                path_to_LST = f'{path_to_lst}{comp_stat}/{year}/'

                # get all LST bands that can be sharped with the S2 composite at this date (and sun angle incidence files as well, as they are dependent on that date
                for k, v in band_dict.items(): # basically, this is a loop over the days 
                    month = v['month']
                    band = int(v['band'])
                    v_path = f'{path_to_LST}Daily_LST_{comp_stat}_{year}_{month}.tif'
                    ds = gdal.Open(v_path, 0)

                    if path_to_dem:
                        # calculate biophysical helper and parameter
                        try:
                            calc_biophys_helper(acq_path=path_to_acq, dem_path=dem_path, lat_path=lat_path, lon_path=lon_path,
                                                vaa_path=path_to_vaa, vza_path=path_to_vza, year=year, month=month,
                                                doy=band, outPath=temp_dump_fold, comp_stat=comp_stat)
                        except Exception as e:
                            print(f'broken because: {e} path_dem={path_to_dem} comp={comp_stat} month={month} day={band} tile={tile_to_process}')
                            break

                        get_biophysical_parameter(path_S2=tilesS2, path_Sun=f"{temp_dump_fold}sun/", year=year, month=month, doy=band,
                                                outPath=temp_dump_fold)
                        
                    # export the LST for that day
                    LST_arr = ds.GetRasterBand(band).ReadAsArray() # store as single Tiff in temp
                    daily_lst_path = f'{temp_dump_fold}Daily_LST_{comp_stat}_{year}_{month}_{band:02d}.tif'
                    makeTif_np_to_matching_tif(LST_arr, v_path, daily_lst_path)
                
                    # store the paths for selecting incidence for corresponding LST
                    incid_date = f'{year}_{month}_{band:02d}.tif'

                    # incidence-tiles
                    # incids = [file for file in getFilelist(path_to_incident, '.tif', deep=True) if tile_to_process in file] 
                    # incid_path = incids[0]
                    incid_path = [file for file in getFilelist(path_to_incident, '.tif', deep=True) if f'{tile_to_process}_{incid_date}' in file][0]

                    # create highRes file through exapnding the vrt of S2
                    highRes_path = f"{temp_dump_fold}HIGHRES_{comp_stat}_{incid_date.split('.')[0]}.vrt"
                    gdal.BuildVRT(highRes_path, [S2_path, slope_path, aspect_path, incid_path], separate=True)

                    for predi in predList: # different predictor combinations for the sharpening
                        if predi == 'allpred':
                            maskVRT_water(highRes_path, colorlist=colors)
                        else:
                            maskVRT_water_and_drop_aux(highRes_path, colorlist=colors)

                        if S2mask == 1:
                            highRes_files.append(f"{highRes_path.split('.')[0]}_watermask.tif")
                            highRes_names.append(f'S2notMasked_{predi}')
                            lowRes_files.append(daily_lst_path)

                        elif S2mask == 2:
                            maskVRT(f"{highRes_path.split('.')[0]}_watermask.tif", mask, suffix=f"_S2_agromask_{predi}")
                            os.remove(f"{highRes_path.split('.')[0]}_watermask.tif")
                            highRes_files.append(f"{highRes_path.split('.')[0]}_watermask_S2_agromask_{predi}.tif")
                            lowRes_files.append(daily_lst_path)
                            highRes_names.append(f"S2Masked_{predi}")

                        elif S2mask == 3:
                            highRes_files.append(f"{highRes_path.split('.')[0]}_watermask.tif")
                            highRes_names.append(f"S2notMasked_{predi}")
                            lowRes_files.append(daily_lst_path)
                            maskVRT(f"{highRes_path.split('.')[0]}_watermask.tif", mask, suffix=f"_S2_agromask_{predi}")
                            highRes_files.append(f"{highRes_path.split('.')[0]}_watermask_S2_agromask_{predi}.tif")
                            lowRes_files.append(daily_lst_path)
                            highRes_names.append(f"S2Masked_{predi}")
            
            # sharp all files for that day and all compstats
            sharpList = []

            for idx, highResFilename in enumerate(highRes_files):
                    lowResFilename = lowRes_files[idx]
                    # f1 = f'{sharp_outFolder}{'/'.join(highResFilename.split('.')[0].split('_')[2:5])}/'
                    for maskname, mask_lowRes in zip(['withoutLSTmask'], ['']): # 'withLSTmask'  lowmask_bin_path
                    # for maskname, mask_lowRes in zip(['withoutLSTmask', 'withLSTmask'], ['', lowmask_bin_path]):
                        lowResMaskFilename = mask_lowRes
                        # f2 = f'{f1}{maskname}/'
                        for movWin in [15]:
                            for cv in [0]:
                                for regrat in [0.25]:
                                    kombi = f'mvwin{movWin}_cv{cv}_regrat{int(regrat*100):02d}_{highRes_names[idx]}_{maskname}'
                                    # f3 = f'{f2}{highRes_names[idx]}/'
                                    # os.makedirs(f'{f3}Residuals/', exist_ok=True)
                                    # os.makedirs(f'{f3}Values/', exist_ok=True)

                                    os.makedirs(f'{sharp_outFolder}Residuals/', exist_ok=True)
                                    os.makedirs(f'{sharp_outFolder}Values/', exist_ok=True)
                                    
                                    # sharpened_file = f'{f3}{'_'.join(highResFilename.split('.')[0].split('_')[2:6])}_{kombi}_{tile_to_process}.tif'
                                    sharpened_file = f"{sharp_outFolder}{'_'.join(highResFilename.split('.')[0].split('_')[1:5])}_{kombi}_{tile_to_process}.tif"
                                    
                                    # check if evap file for sharpened_file already exists (might happen when we restart a process that terminated unexpectedly
                                    # before and now we go through the last date from datelist that had been processed)
                                    file_name = f"{evap_outFolder}{sharpened_file.split('/')[-1].split('_')[0]}_{sharpened_file.split('/')[-1].split('_')[1]}\
                                        _{sharpened_file.split('/')[-1].split('_')[2]}_{int(sharpened_file.split('/')[-1].split('_')[3])}_{movWin}_{cv}_{regrat}_\
                                            {sharpened_file.split('/')[-1].split('_')[9]}_{sharpened_file.split('/')[-1].split('_')[7]}_\
                                                {sharpened_file.split('/')[-1].split('_')[8]}_{tile_to_process}"
                                    

                                    if (os.path.exists(f"{file_name}_ET_Canopy_func.tif") and os.path.exists(f"{file_name}_ET_Soil_func.tif")):
                                        print(f"{file_name} already created")
                                        continue

                                    runSharpi(highResFilename, lowResFilename, lowResMaskFilename, cv, movWin, regrat, sharpened_file, useDecisionTree = True)
                                    
                                    # run the biophysical parameter calculation here
                                    
                                    # should run the function and return a list with the paths for all the created files
                                    # this list will then be passed on to runEvapi
                                    sharpList.append(sharpened_file)
            
            # calc evapo for all compstats at that day
            for sharped in sharpList:
                comp = sharped.split('/')[-1].split('_')[0]
                year = sharped.split('/')[-1].split('_')[1]
                month = sharped.split('/')[-1].split('_')[2]
                day = int(sharped.split('/')[-1].split('_')[3])
                mvwin = sharped.split('/')[-1].split('_')[4]
                cv = sharped.split('/')[-1].split('_')[5]
                regrat = sharped.split('/')[-1].split('_')[6]
                sharp = sharped.split('/')[-1].split('_')[8]
                s2Mask = sharped.split('/')[-1].split('_')[7]
                lstMask = sharped.split('/')[-1].split('_')[9]
                tile = '_'.join(sharped.split('/')[-1].split('.')[0].split('_')[-2:])

                if path_to_dem:
                    bio_pars = [file for file in getFilelist(f'{temp_dump_fold}bio/', '.tif') if f'{year}_{month}_{day}.tif' in file]
                else:
                    bio_pars = False

                

                runEvapi(year=year, month=month, day=day, comp=comp, sharp=sharp, s2Mask=s2Mask, lstMask=lstMask, tile=tile,
                        tempDir=trash_path, path_to_temp=temp_dump_fold, path_to_sharp=sharp_outFolder, mvwin=mvwin, cv=cv,
                        regrat=regrat, evap_outFolder=evap_outFolder, S2path=S2_path, printInterim=printEvapInter, bio=bio_pars)
            
            
            # at the end of the date loop --> clean up temp folder and sharp
            temp_files_vrt = [file for file in getFilelist(temp_dump_fold, '.vrt')]
            temp_files_tif = [file for file in getFilelist(temp_dump_fold, '.tif')]

            if path_to_dem:
                [temp_files_vrt.remove(path) for path in [slope_path, aspect_path, thuenen_path, dem_path, lat_path, lon_path]]
                bio_files = [file for file in getFilelist(f'{temp_dump_fold}bio/', '.tif')]
                sun_files = [file for file in getFilelist(f'{temp_dump_fold}sun/', '.tif')]
                [os.remove(file) for file in bio_files]
                [os.remove(file) for file in sun_files]

            else:    
                [temp_files_vrt.remove(path) for path in [slope_path, aspect_path, thuenen_path]]
            sharp_files = [file for file in getFilelist(sharp_outFolder, '.tif', deep=True)]
            [os.remove(file) for file in temp_files_vrt]
            [os.remove(file) for file in temp_files_tif]
            [os.remove(file) for file in sharp_files]

        
            # export the processed dates
            df = pd.DataFrame({
                'compdates': compDate,
                })
            df.to_csv(compdate_path, index=False)

    else:
        print('JOB ABORTED! NO VALID PARALLEL MODE PROVIDED!')


def calc_biophys_helper(acq_path, dem_path, lat_path, lon_path, vaa_path, vza_path, year, month, doy, outPath, comp_stat):
    # find S3 acquisition file and warp the respecitve day
    acq_time_file = [file for file in getFilelist(f'{acq_path}{year}', '.tif') \
                 if not any(substr in file for substr in ['readable', 'minVZA', 'order']) and
                 month in file][0]
    warped_ds = warp_raster_to_reference(acq_time_file, reference_path=dem_path, output_path='MEM', resampling='near')
    time_warp = warped_ds.GetRasterBand(doy).ReadAsArray()

    # get topo data
    lat = stackReader(lat_path)
    lon = stackReader(lon_path)
    dem = stackReader(dem_path)

    # Flatten Unix time and convert to datetime
    timestamps_flat = pd.to_datetime(time_warp.ravel(), unit='s', utc=True)

    # Flatten lat/lon and DEM
    lat_flat = lat.ravel()
    lon_flat = lon.ravel()
    dem_flat = dem.ravel()

    # calculate and export zenith and azimuth per pixel for time of observation
    solpos = solarposition.get_solarposition(time=timestamps_flat, latitude=lat_flat, longitude=lon_flat, altitude=dem_flat)
    zen = solpos['zenith'].values.reshape(time_warp.shape)
    azi = solpos['azimuth'].values.reshape(time_warp.shape)
    zen[zen>=90] = np.nan

    npTOdisk(zen, dem_path,
             path_safe(f'{outPath}sun/sun_zenith_degrees_{year}_{month}_{doy}.tif'), bandnames='July_9_2019', noData=-32768)
    npTOdisk(azi, dem_path,
             path_safe(f'{outPath}sun/sun_azimuth_degrees_{year}_{month}_{doy}.tif'), bandnames='July_9_2019', noData=-32768)
    
    # find and export sensor zenith and azimuth
    vza_ds = gdal.Open([file for file in getFilelist(vza_path, '.tif', deep=True) if all(substr in file for substr in [year, month, comp_stat])][0])
    vza_warped = warp_raster_to_reference(vza_ds, dem_path, output_path='MEM', resampling='nearest')
    vza_warped_day = vza_warped.GetRasterBand(doy).ReadAsArray()
    npTOdisk(vza_warped_day, dem_path, path_safe(f'{outPath}sun/sensor_zenith_degrees_{year}_{month}_{doy}.tif'),
             bandnames=f'{month}_{doy}_{year}', noData=-32768)

    vaa_ds = gdal.Open([file for file in getFilelist(vaa_path, '.tif', deep=True) if all(substr in file for substr in [year, month, comp_stat])][0])
    vaa_warped = warp_raster_to_reference(vaa_ds, dem_path, output_path='MEM', resampling='nearest')
    vaa_warped_day = vaa_warped.GetRasterBand(doy).ReadAsArray()
    npTOdisk(vaa_warped_day, dem_path, path_safe(f'{outPath}sun/sensor_azimuth_degrees_{year}_{month}_{doy}.tif'),
             bandnames=f'{month}_{doy}_{year}', noData=-32768)


def get_biophysical_parameter(path_S2, path_Sun, year, month, doy, outPath):
    varNames = ['Albedo', 'fAPAR', 'fCOVER', 'LAI', 'CCC', 'CWC']
    s2 = read_s2_force(s2_dir=path_S2, sun_dir=path_Sun, year=year, month=month, doy=doy)

    # calculate per variable
    for varName in varNames:
        sl2p_inp = SL2P.prepare_sl2p_inp(s2, varName, 'S2_FORCE')
        varmap=SL2P.SL2P(sl2p_inp, varName, 'S2_FORCE')

        # output_dir = r'C:\Users\Leoun\Work\SL2P-FORCE-main\output\new'
        # tile_basename = os.path.basename(os.path.normpath(tile_dir))
        # imageName = os.path.join(output_dir, tile_basename) # Create a base name to replace the SAFE filename

        profile = s2['profile'].copy()
        profile.update({
            'count': 4, 
            'dtype': rasterio.float32, # Use float32 for all 4 bands
            'driver': 'GTiff'
        }) 

        flag_inp = varmap['sl2p_inputFlag'].astype(numpy.float32)
        flag_out = varmap['sl2p_outputFlag'].astype(numpy.float32)

        outP = path_safe(f'{outPath}bio/{varName}_{year}_{month}_{doy}.tif')

        # 4. Write all 4 layers
        with rasterio.open(outP, 'w', **profile) as dst:
            dst.write(varmap[varName].astype(numpy.float32), 1)
            dst.write(varmap[f'{varName}_uncertainty'].astype(numpy.float32), 2)
            dst.write(flag_inp, 3) # Write the converted input flag
            dst.write(flag_out, 4)  # Write the converted output flag


def read_biophys(list_of_paths):
    conti = []
    for par in ['Albedo', 'CCC', 'CWC', 'LAI', 'fAPAR', 'fCOVER']:
        for path in list_of_paths:
            if par in path:
                arr = stackReader(path)
                conti.append(arr[:,:,0])
    return conti[0], conti[1], conti[2], conti[3], conti[4], conti[5]


def compute_fg_fipar_pai(LAI, FAPAR, theta, max_iter=20, tol=1e-4):
    """
    Vectorized implementation of Equations, following Guzinski et al 2020
    """

    # Initial conditions
    fg = np.ones_like(LAI)
    PAI = LAI.copy()

    cos_theta = np.cos(theta)
    cos_theta = np.clip(cos_theta, 0.01, 1.0)

    for _ in range(max_iter):
        # Eq. (8)
        FIPAR = 1.0 - np.exp(-0.5 * PAI / cos_theta)

        # Avoid division by zero
        FIPAR = np.clip(FIPAR, 1e-6, 1.0)

        # Eq. (7)
        fg_new = FAPAR / FIPAR
        fg_new = np.clip(fg_new, 0.01, 1.0)

        # Check convergence
        if np.nanmax(np.abs(fg_new - fg)) < tol:
            break

        fg = fg_new
        # Eq. (9)
        PAI = LAI / fg

    return fg, FIPAR, PAI


def leaf_optics_vis(Cab):
    """
    Compute visible leaf reflectance and transmittance.
    Cab: chlorophyll content [g/m²] or μg/cm²
    """
    # Clip to avoid numerical overflow
    Cab_safe = np.clip(Cab, 0, 200)
    
    rho = 0.05 + 0.45 * np.exp(-0.015 * Cab_safe)
    tau = 0.05 + 0.35 * np.exp(-0.020 * Cab_safe)

    # Ensure no NaN or inf
    rho = np.nan_to_num(rho, nan=0.05, posinf=0.5, neginf=0.05)
    tau = np.nan_to_num(tau, nan=0.05, posinf=0.4, neginf=0.05)

    # Clip physically valid range [0,1]
    rho = np.clip(rho, 0.0, 1.0)
    tau = np.clip(tau, 0.0, 1.0)

    return rho, tau


def leaf_optics_nir(Cw):
    """
    Compute NIR leaf reflectance and transmittance.
    Cw: equivalent water thickness [mm]
    """
    Cw_safe = np.clip(Cw, 0, 0.1)

    rho = 0.40 + 0.30 * np.exp(-20.0 * Cw_safe)
    tau = 0.30 + 0.20 * np.exp(-25.0 * Cw_safe)

    rho = np.nan_to_num(rho, nan=0.40, posinf=0.7, neginf=0.40)
    tau = np.nan_to_num(tau, nan=0.30, posinf=0.5, neginf=0.30)

    rho = np.clip(rho, 0.0, 1.0)
    tau = np.clip(tau, 0.0, 1.0)

    return rho, tau


def make_ET_median(vrt_folder, ET_var, date_s, date_e, reg_search, outFolder, mask=False):
    
    files = getFilelist(f'{vrt_folder}{ET_var}/', '.vrt')
    dicti = defaultdict(list)

    for file in files:
            m = reg_search.search(file).group()[:-1]
            year, month, day = m.split('_')
            day = day.zfill(2)  # "1" → "01", "13" → "13"
            date_key = f"{year}{MONTH_TO_02D[month]}{day}"
            
            if date_s <= datetime.strptime(date_key, '%Y%m%d') < date_e:
                dicti[datetime.strftime(date_s, '%Y_%m_%d')].append(file)
        
    for k, v in dicti.items():
        arrL = []
        for file in v:
            ds = gdal.Open(file)
            arr = ds.GetRasterBand(1).ReadAsArray()
            arr[arr<=0] = np.nan
            arr[arr>12] = np.nan
            arrL.append(arr)
        median_arr = np.nanmedian(np.dstack(arrL),axis=2)

        if mask is not False:
             outarr = median_arr * mask
        else:
             outarr = median_arr
        makeTif_np_to_matching_tif(outarr, file, f"{outFolder}{ET_var}_{k}_median_ET.tif", 0)