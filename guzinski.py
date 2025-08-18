from helperToolz.helpsters import is_leap_year, checkPath
from helperToolz.dicts_and_lists import *
import numpy as np
from osgeo import gdal, osr
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
        print('temperature will be processed before surface pressure - continue')
    else:
        cont = list_of_era5_variables[press_ind]
        list_of_era5_variables[press_ind] = list_of_era5_variables[temp_ind]
        list_of_era5_variables[temp_ind] = cont
        print('2m_temperature and surface pressure swaped and now in right order - continue')

 
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
                           sharp_blendheight=False, sharp_DEM=False, sharp_geopot=False, sharp_rate=False, sharp_temp=False, sharpener=False):
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
                                        sharp_geopot=sharp_geopot, sharp_rate=sharp_rate, sharp_temp=sharp_temp, sharpener=sharpener)
    else:
        warp_ERA5_to_reference(grib_path=path_to_era_grib, reference_path=reference_path, resampling=resampling,
                               output_path=outPath, bandL=bandL,
                               sharp_blendheight=sharp_blendheight, sharp_DEM=sharp_DEM,
                               sharp_geopot=sharp_geopot, sharp_rate=sharp_rate, sharp_temp=sharp_temp, sharpener=sharpener)
        era_ds = gdal.Open(outPath)
    
    bandNumber = era_ds.RasterCount
    era_time = [pd.Timestamp(era_ds.GetRasterBand(i+1).GetDescription()) for i in range(bandNumber)]

    # open and load the acquisition raster from LST
    if isinstance(lst_acq_file, str):
        LST_acq_ds = gdal.Open(lst_acq_file)
    else:
        LST_acq_ds = lst_acq_file
    # her should be a loop if we go for more than one day
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
                                           aspect_path, dem_path, lat_path, lon_path, outPath='MEM', bandL='ALL'):
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
        era_ds = warp_ERA5_to_reference(grib_path=path_to_ssrdsc_grib, reference_path=reference_path, output_path=outPath)
    else:
        warp_ERA5_to_reference(grib_path=path_to_ssrdsc_grib, reference_path=reference_path, output_path=outPath)
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

    return poa_global_arr, zenith_arr, azimuth_arr, ssrd_watt # *3600 to bring back to J/m²
    


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

