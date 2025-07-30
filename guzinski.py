from helperToolz.helpsters import is_leap_year
from helperToolz.dicts_and_lists import *
import numpy as np
from osgeo import gdal
from datetime import datetime, timezone

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
            'band': day_in_month
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
    """Warps a ERA .gib file to an existing raster in terms of resolution, projection, extent and aligns to raster

    Args:
        grib_path (str): complete path to the grib file
        reference_path (str): complete path to the DEM raster to which the grib dataset will be warped to
        output_path (str): path to filename where the raster stacks will be stored.
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
        print('DEM Sharpener will bei applied')
         # load sharpener datasets
        dem_ds = gdal.Open(sharp_DEM)
        dem = dem_ds.GetRasterBand(1).ReadAsArray()
        geopot_ds = gdal.Open(sharp_geopot)
        geopot = geopot_ds.GetRasterBand(1).ReadAsArray()
        
        if not single:
            bandCount = warped_ds.RasterCount
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

    elif any([sharp_DEM, sharp_geopot, sharp_blendheight, sharp_rate]):
        raise ValueError('Not all datasets for sharpening provided - Please provide all or none sharpener parameter')
    
    else:
        if not single:
            bandCount = warped_ds.RasterCount
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
