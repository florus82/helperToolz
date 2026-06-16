import datetime as dt
from osgeo import gdal, osr
gtiff_driver = gdal.GetDriverByName('GTiff')
import sys
sys.path.append('/home/potzschf/repos/')
from helperToolz.helpsters import *
from helperToolz.dicts_and_lists import *
from helperToolz.guzinski import * 
from helperToolz.mirmazloumi import *


class LandsatETFileManager:
    '''
    takes a list of paths to landsat provisional ET files, and helps to extract files filtered by a date, a date range, or by year, month, or day
    get_by_date: provide a date (e.g. datetime.date(2022,6,25))
    get_by_range: provide a starting and ending date (both dates will be included in the output)
    get_by_year: provide a year (int)
    get_by_month: provide a month (int, e.g. January = 1)
    get_by_day: provide a day (int)
    get_by_year_and_month: provide as 2 int
    get_all_dates: returns all dates of the initialiszed class. If no parameter provided, unsorted list will be returned. If boolean True provided, sorted list will be returned
    get_all_dates_by_year: provide a year, and all dates are returned (unique=T/F)
    '''
    def __init__(self, landsat_files):
        self.date_to_files = {}

        self._build_index(landsat_files)
    # internal function that fills dict when class is initialized
    def _build_index(self, landsat_files):
        for file in landsat_files:
            filename = file.split('/')[-1]
            year = int(filename[10:14])
            month = int(filename[14:16])
            day = int(filename[16:18])
            print(f"{year}-{month}-{day}")
            date = dt.date(year, month, day)

            if date not in self.date_to_files:
                self.date_to_files[date] = []
            self.date_to_files[date].append(file)
    
    def get_by_date(self, date):
        return self.date_to_files.get(date, [])

    def get_by_range(self, start_date, end_date):
        matching_files = []
        for date, files in self.date_to_files.items():
            if start_date <= date <= end_date:
                for file in files:
                    matching_files.append((date, file))
        matching_files.sort()
        return [file for date, file in matching_files]
    
    def get_by_year(self, year):
        matching_files = []
        for date, files in self.date_to_files.items():
            if date.year == year: # works because it is a datetime object
                for file in files:
                    matching_files.append((date, file))
        matching_files.sort()
        return [file for date, file in matching_files]
    
    def get_by_month(self, month):
        matching_files = []
        for date, files in self.date_to_files.items():
            if date.month == month: # works because it is a datetime object
                for file in files:
                    matching_files.append((date, file))
        matching_files.sort()
        return [file for date, file in matching_files]
    
    def get_by_day(self, day):
        matching_files = []
        for date, files in self.date_to_files.items():
            if date.day == day: # works because it is a datetime object
                for file in files:
                    matching_files.append((date, file))
        matching_files.sort()
        return [file for date, file in matching_files]
        
    def get_all_dates(self, sorted_output=False):
        dates = self.date_to_files.keys()
        return sorted(dates) if sorted_output else list(dates)
    
    def get_all_dates_by_year(self, year, unique=False):
        matching_files = []
        for date, files in self.date_to_files.items():
            if date.year == year: # works because it is a datetime object
                for file in files:
                    matching_files.append((date, file))
        matching_files.sort()
        dates = [date for date, file in matching_files]
        if unique:
            dates  = list(dict.fromkeys(dates))
        return dates
    
    def get_all_dates_by_year_and_month(self, year, month, unique=False):
        matching_files = []
        for date, files in self.date_to_files.items():
            if date.year == year and date.month == month: # works because it is a datetime object
                for file in files:
                    matching_files.append((date, file))
        matching_files.sort()
        dates = [date for date, file in matching_files]
        if unique:
            dates  = list(dict.fromkeys(dates))
        return dates
    
    def get_by_year_and_month(self, year, month):
        matching_files = []
        for date, files in self.date_to_files.items():
            if date.year == year and date.month == month: # works because it is a datetime object
                for file in files:
                    matching_files.append((date, file))
        matching_files.sort()
        return [file for date, file in matching_files]
    

def xarray_to_gdal_mem(xr_da):
    
    # Apply scaling and cast to desired integer type
    data = xr_da.values
    height, width = data.shape

    # Auto-detect spatial dimensions
    dim_names = list(xr_da.dims)
    y_dim = [d for d in dim_names if 'y' in d.lower() or 'lat' in d.lower()][0]
    x_dim = [d for d in dim_names if 'x' in d.lower() or 'lon' in d.lower()][0]

    x_coords = xr_da.coords[x_dim].values
    y_coords = xr_da.coords[y_dim].values

    # Ensure ascending order and consistent resolution
    xres = float(x_coords[1] - x_coords[0])
    yres = float(y_coords[1] - y_coords[0])
    origin_x = float(x_coords[0])
    origin_y = float(y_coords[0])
    
    transform = [origin_x, xres, 0, origin_y, 0, yres]

    # Create in-memory GDAL dataset
    driver = gdal.GetDriverByName('MEM')
    ds = driver.Create('', width, height, 1, gdal.GDT_Float32)
    ds.GetRasterBand(1).WriteArray(data)
    ds.SetGeoTransform(transform)

    # Set projection if available
    if hasattr(xr_da, 'rio') and xr_da.rio.crs:
        srs = osr.SpatialReference()
        srs.ImportFromWkt(xr_da.rio.crs.to_wkt())
        ds.SetProjection(srs.ExportToWkt())
    return ds

def warp_to_template(source_ds, reference_path, outPath=None, mask_path=None, resampling='bilinear', outType=gdal.GDT_UInt16):
    '''
    source_ds: a gdal.Open() object that will be warped
    reference_path: the input raster_ds will be warped to the raster stored at thils location
    outPath: if provided, raster will be exported as tiff to this location
    mask_path: path to the raster which will be used as mask (should have the same extent/resolution as raster at reference path). WORKS only if outPath provided
    resampling: method for resampling when warping, e.g. nearest, cubic (default 'bilinear')
    outType: gdal object can be provided to set datatype of output (default gdal.GDT_UInt16)
    '''
    # Open reference raster
    ref_ds = gdal.Open(reference_path)
    ref_proj = ref_ds.GetProjection()
    ref_gt = ref_ds.GetGeoTransform()
    x_size = ref_ds.RasterXSize
    y_size = ref_ds.RasterYSize

    # Calculate bounds
    xmin = ref_gt[0]
    ymax = ref_gt[3]
    xres = ref_gt[1]
    yres = abs(ref_gt[5])
    xmax = xmin + xres * x_size
    ymin = ymax - yres * y_size

    # Define in-memory output path
    mem_path = '/vsimem/warped.tif'

    # Set up warp options
    warp_options = gdal.WarpOptions(
        format='GTiff',
        dstSRS=ref_proj,
        outputBounds=(xmin, ymin, xmax, ymax),
        xRes=xres,
        yRes=yres,
        resampleAlg=resampling,
        outputType=outType
    )

    # Perform reprojection and resampling
    gdal.Warp(mem_path, source_ds, options=warp_options)

    if outPath:
        if mask_path:
             # Open the in-memory file
            warped_ds = gdal.Open(mem_path)
            warp_arr = warped_ds.GetRasterBand(1).ReadAsArray()
            mask_ds = gdal.Open(mask_path)
            mask_arr = mask_ds.GetRasterBand(1).ReadAsArray()
            warped_masked = warp_arr * mask_arr
            target_ds = gtiff_driver.Create(outPath, warped_masked.shape[1] , warped_masked.shape[0], 1, outType)           
            target_ds.SetGeoTransform(warped_ds.GetGeoTransform())
            target_ds.SetProjection(warped_ds.GetProjection())
            band = target_ds.GetRasterBand(1)
            #band.SetNoDataValue(0)
            band.WriteArray(warped_masked)
            del target_ds

        else:
            gdal.Warp(outPath, source_ds, options=warp_options)
    else:
        # Open the in-memory file
        warped_ds = gdal.Open(mem_path)
        warped_arr = warped_ds.GetRasterBand(1).ReadAsArray()
        if not mask_path:
            return warped_arr
        else:
            mask_ds = gdal.Open(mask_path)
            mask_arr = mask_ds.GetRasterBand(1).ReadAsArray()
            warped_masked = warped_arr * mask_arr
            return warped_masked
            

def Sharp_Evap_Sensi(tile_to_process, storFolder, path_to_slope, path_to_aspect, path_to_agro, path_to_force,
               path_to_lst, time_start, time_end, compList, predList, S2mask, printEvapInter=False, # path_to_inci, 
               path_to_dem=False, path_to_lat=False, path_to_lon=False, path_to_acq=False, path_to_vaa=False, path_to_vza=False, 
               para_mode='perTILE', compDate=False, movWinL=False, cvL=False, regratL=False):

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
    thuenen = [file for file in getFilelist(path_to_agro, '.tif') if tile_to_process in file and year in file and '_bin_' not in file] # if any tile name is in file

    # get those tiles (and composite if more than one tile is provided)
    if compDate: # if parallel over Dates, problem with multiple access on vrts might occur --> different vrts per compDate
        slope_path = f"{temp_dump_fold}SLOPE_{year}_{compDate}.vrt"
        thuenen_path = f"{temp_dump_fold}THUENEN_{year}_{compDate}.vrt"
        aspect_path = f"{temp_dump_fold}ASPECT_{year}_{compDate}.vrt"
    else:
        slope_path = f"{temp_dump_fold}SLOPE_{year}.vrt"
        thuenen_path = f"{temp_dump_fold}THUENEN_{year}.vrt"
        aspect_path = f"{temp_dump_fold}ASPECT_{year}.vrt"

    gdal.BuildVRT(slope_path, slopes)
    gdal.BuildVRT(aspect_path, aspects)
    gdal.BuildVRT(thuenen_path, thuenen)

        # needed to estimate biophysical parameter
    if path_to_dem:
        # dem-tiles
        dems = [file for file in getFilelist(path_to_dem, '.tif') if tile_to_process in file] # if any tile name is in file
        # lat-tiles
        lats = [file for file in getFilelist(path_to_lat, '.tif') if tile_to_process in file] # if any tile name is in file
        # lon-tiles
        lons = [file for file in getFilelist(path_to_lon, '.tif') if tile_to_process in file] # if any tile name is in file

        if compDate:
            dem_path = f'{temp_dump_fold}DEM_{year}_{compDate}.vrt'
            lat_path = f'{temp_dump_fold}LAT_{year}_{compDate}.vrt'
            lon_path = f'{temp_dump_fold}LON_{year}_{compDate}.vrt'
        else:
            dem_path = f'{temp_dump_fold}DEM_{year}.vrt'
            lat_path = f'{temp_dump_fold}LAT_{year}.vrt'
            lon_path = f'{temp_dump_fold}LON_{year}.vrt'
        
        gdal.BuildVRT(dem_path, dems)
        gdal.BuildVRT(lat_path, lats)
        gdal.BuildVRT(lon_path, lons)


    # make the mask ready for S2 masking
    th_ds = gdal.Open(thuenen_path)
    th_arr = th_ds.GetRasterBand(1).ReadAsArray()
    mask = np.where(th_arr == -9999, 0, 1)

    if year in ['2018', '2022']:
        colors = ['BLU', 'GRN', 'RED', 'BNR', 'NIR', 'RE1', 'RE2', 'RE3',  'SW1', 'SW2']
    else:
        colors = ['BLUE', 'GREEN', 'RED',  'BROADNIR', 'NIR', 'REDEDGE1', 'REDEDGE2', 'REDEDGE3', 'SWIR1', 'SWIR2']

    # ################ load force and vrt
    path_to_S2_tiles = f'{path_to_force}/{year}/'
        
    if para_mode == 'perDATE':
        if compDate == False:
            print('NO SINGLE DATE PROVIDED - THIS PROCESS WILL CRASH SOON :(')
        
        compdate_path = f"{evap_outFolder}compdates_{compDate}.csv"

        if int(time_start) <= int(compDate) <= int(time_end): # neede to be adapted in a manner that incorporates the +/-4 otherwise time_start 20190401 
            # would find date 20190406 as int(time_start) <= int(date) <= int(time_end)..

            lowRes_files = []
            highRes_files = []
            highRes_names = []
            sharpList = []

            # get S2 bands
            tilesS2 = [file for file in getFilelist(path_to_S2_tiles, '.tif', deep=True) if tile_to_process in file and f'{compDate}.tif' in file]
            tilesS2 = [t2 for col in colors for t2 in tilesS2 if f"_{col}_" in t2]
            S2_path = f'{temp_dump_fold}S2_{compDate}.vrt'
            
            # build vrt
            tmp_vrt = f"{S2_path}.tmp"
            vrt = gdal.BuildVRT(tmp_vrt, tilesS2, separate=True)
            vrt = None
            os.rename(tmp_vrt, S2_path)   # atomic operation --> this ensures, that vrt fully written before accessed (parallel crap)
            # vrt = gdal.BuildVRT(S2_path, tilesS2, separate=True)
            # vrt = None
            vrt = gdal.Open(S2_path, gdal.GA_Update)  # VRT must be writable
            for idz, bname in enumerate(colors): 
                band = vrt.GetRasterBand(1+idz)
                band.SetDescription(bname)
            vrt = None


            # determine LST and incidence files associated with respective S2 composite
            band_dict = transform_compositeDate_into_LSTbands(compDate, 4)

            # stat used for compositing
            for comp_stat in compList: #  
                path_to_LST = f'{path_to_lst}{comp_stat}/{year}/'

                # get all LST bands that can be sharped with the S2 composite at this date (and sun angle incidence files as well, as they are dependent on that date
                for k, v in band_dict.items(): # basically, this is a loop over the days 
                    month = v['month']
                    band = int(v['band'])

                    # export the LST for that day
                    daily_lst_path = f'{temp_dump_fold}Daily_LST_{comp_stat}_{year}_{month}_{band:02d}.tif'
                    if os.path.exists(daily_lst_path):
                        pass
                    else:
                        v_path = f'{path_to_LST}Daily_LST_{comp_stat}_{year}_{month}.tif'
                        ds = gdal.Open(v_path, 0)
                        LST_arr = ds.GetRasterBand(band).ReadAsArray() # store as single Tiff in temp
                        makeTif_np_to_matching_tif(LST_arr, v_path, daily_lst_path)


                    # create highRes file through exapnding the vrt of S2
                    highRes_path = f"{temp_dump_fold}HIGHRES_{comp_stat}_{year}_{month}_{band:02d}.vrt"

                    if movWinL:
                        # create combination lists of 3 parameter for easier looping (with this if else condition here)
                        combos = [(mov, cv, reg) for mov in movWinL for cv in cvL for reg in regratL]
                        movWinL, cvL, regratL = map(list, zip(*combos))
                    else:
                        movWinL = [15]
                        cvL = [0]
                        regratL = [25]

                    for movWin, cv, regrat in zip(movWinL, cvL, regratL):
                            # first check if already sharpened or evaped
                            skipper_counter = 0
                            for predi in predList: # different predictor combinations for the sharpening
                                skip_pred = 0
                                skip_sharp = 0
                                if S2mask == 1:
                                    expected_evap_files = [
                                        f"{evap_outFolder}{comp_stat}_{year}_{month}_{band}_mvwin{movWin}_cv{cv}_regrat{regrat}_withoutLSTmask_S2notMasked_{predi}_{tile_to_process}_ET_Canopy_func.tif",
                                        f"{evap_outFolder}{comp_stat}_{year}_{month}_{band}_mvwin{movWin}_cv{cv}_regrat{regrat}_withoutLSTmask_S2notMasked_{predi}_{tile_to_process}_ET_Soil_func.tif"
                                    ]
                                    expected_sharp_files = [
                                        f"{sharp_outFolder}Values/{comp_stat}_{year}_{month}_{band:02d}_mvwin{movWin}_cv{cv}_regrat{regrat}_withoutLSTmask_S2notMasked_{predi}_{tile_to_process}.tif"
                                    ]
                                elif S2mask == 2:
                                    expected_evap_files = [
                                        f"{evap_outFolder}{comp_stat}_{year}_{month}_{band}_mvwin{movWin}_cv{cv}_regrat{regrat}_withoutLSTmask_S2Masked_{predi}_{tile_to_process}_ET_Canopy_func.tif",
                                        f"{evap_outFolder}{comp_stat}_{year}_{month}_{band}_mvwin{movWin}_cv{cv}_regrat{regrat}_withoutLSTmask_S2Masked_{predi}_{tile_to_process}_ET_Soil_func.tif"
                                    ]
                                    expected_sharp_files = [
                                        f"{sharp_outFolder}Values/{comp_stat}_{year}_{month}_{band:02d}_mvwin{movWin}_cv{cv}_regrat{regrat}_withoutLSTmask_S2Masked_{predi}_{tile_to_process}.tif"
                                    ]
                                elif S2mask == 3:
                                    expected_evap_files = [
                                        f"{evap_outFolder}{comp_stat}_{year}_{month}_{band}_mvwin{movWin}_cv{cv}_regrat{regrat}_withoutLSTmask_S2Masked_{predi}_{tile_to_process}_ET_Canopy_func.tif",
                                        f"{evap_outFolder}{comp_stat}_{year}_{month}_{band}_mvwin{movWin}_cv{cv}_regrat{regrat}_withoutLSTmask_S2Masked_{predi}_{tile_to_process}_ET_Soil_func.tif",
                                        f"{evap_outFolder}{comp_stat}_{year}_{month}_{band}_mvwin{movWin}_cv{cv}_regrat{regrat}_withoutLSTmask_S2notMasked_{predi}_{tile_to_process}_ET_Canopy_func.tif",
                                        f"{evap_outFolder}{comp_stat}_{year}_{month}_{band}_mvwin{movWin}_cv{cv}_regrat{regrat}_withoutLSTmask_S2notMasked_{predi}_{tile_to_process}_ET_Soil_func.tif"
                                    ]
                                    expected_sharp_files = [
                                        f"{sharp_outFolder}Values/{comp_stat}_{year}_{month}_{band:02d}_mvwin{movWin}_cv{cv}_regrat{regrat}_withoutLSTmask_S2Masked_{predi}_{tile_to_process}.tif",
                                        f"{sharp_outFolder}Values/{comp_stat}_{year}_{month}_{band:02d}_mvwin{movWin}_cv{cv}_regrat{regrat}_withoutLSTmask_S2notMasked_{predi}_{tile_to_process}.tif"
                                    ]

                                # if all expected evapo estimates are there, skip this file
                                if all(os.path.exists(path) for path in expected_evap_files):
                                    print('combi already processed', flush=True)
                                    skip_pred = 1
                                    skipper_counter += 1

                                # if there is a problem with computing this day, skip this file
                                elif any(os.path.exists(errorlog) for errorlog in 
                                            [f"{trash_path}ERROR_{comp_stat}_{year}_{month}_{band}_mvwin{movWin}_cv{cv}_regrat{regrat}_withoutLSTmask_S2notMasked_{predi}_{tile_to_process}_ET.log",
                                            f"{trash_path}ERROR_{comp_stat}_{year}_{month}_{band}_mvwin{movWin}_cv{cv}_regrat{regrat}_withoutLSTmask_S2Masked_{predi}_{tile_to_process}_ET.log"]):
                                    print('evaping this day not possible')
                                    skip_pred = 1
                                    skipper_counter += 1

                                # if not all evapo estimates exist, but all sharpened files, transfer sharpened files into queue for processing without sharpening again
                                else:
                                    if all(os.path.exists(path) for path in expected_sharp_files):
                                        print('combi already shapred - pass on to evap queue', flush=True)
                                        for sharpedFile in expected_sharp_files:
                                            sharpList.append(sharpedFile)
                                            print(sharpedFile)
                                        skip_sharp = 1
                            
                                if skip_sharp or skip_pred == 1:
                                    continue
                                if predi == 'allpred':
                                    # calculate incidence    
                                    calc_Incidence(tile=tile_to_process, year=year, comp=comp_stat, outFolder=temp_dump_fold, time_dict=band_dict) 
                                    incid_path = f'{temp_dump_fold}INCIDENCE_{comp_stat}_{year}_{month}_{band:02d}.tif'
                                    gdal.BuildVRT(highRes_path, [S2_path, slope_path, aspect_path, incid_path], separate=True)
                                    maskVRT_water(highRes_path, colorlist=colors)
                                else:
                                    gdal.BuildVRT(highRes_path, [S2_path], separate=True)
                                    maskVRT_water(highRes_path, colorlist=colors)
                                    # maskVRT_water_and_drop_aux(highRes_path, colorlist=colors)

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
                            if skipper_counter != len(predList): # i.e. there are evap estimations in the pipeline
                                if path_to_dem:

                                    # check if bio products are already there
                                    if os.path.exists(f"{temp_dump_fold}bio/"):
                                        if len([bfile for bfile in getFilelist(f"{temp_dump_fold}bio/", '.tif') if f"{comp_stat}_{year}_{month}_{band:02d}" in bfile]) == 6:
                                            bio = 1
                                        else:
                                            bio = 0
                                    else:
                                        bio = 0

                                    # if bio is not there, check if sun already there
                                    if bio == 0:
                                        if os.path.exists(f"{temp_dump_fold}sun/"):
                                            if len([sfile for sfile in getFilelist(f"{temp_dump_fold}sun/", '.tif') if f"{comp_stat}_{year}_{month}_{band:02d}" in sfile]) == 4:
                                                sun = 1
                                            else:
                                                sun = 0
                                        # if sun not there, calc it           
                                        if sun == 0: # 
                                            # calculate biophysical helper and parameter
                                            try:
                                                calc_biophys_helper(acq_path=path_to_acq, dem_path=dem_path, lat_path=lat_path, lon_path=lon_path,
                                                                    vaa_path=path_to_vaa, vza_path=path_to_vza, year=year, month=month,
                                                                    doy=band, outPath=temp_dump_fold, comp_stat=comp_stat)
                                            except Exception as e:
                                                print(f'broken because: {e} path_dem={path_to_dem} comp={comp_stat} month={month} day={band} tile={tile_to_process}')
                                                #break
                                        # calc bio
                                        get_biophysical_parameter(path_S2=tilesS2, path_Sun=f"{temp_dump_fold}sun/", year=year, month=month, doy=f"{band:02d}",
                                                                outPath=temp_dump_fold, comp_stat=comp_stat)
            if len(highRes_files) != 0:
                for idx, highResFilename in enumerate(highRes_files):
                    lowResFilename = lowRes_files[idx]
                    for maskname, mask_lowRes in zip(['withoutLSTmask'], ['']): # 'withLSTmask'  lowmask_bin_path
                    # for maskname, mask_lowRes in zip(['withoutLSTmask', 'withLSTmask'], ['', lowmask_bin_path]):
                        lowResMaskFilename = mask_lowRes
                        kombi = f'mvwin{movWin}_cv{cv}_regrat{regrat}_{highRes_names[idx]}_{maskname}'

                        os.makedirs(f"{sharp_outFolder}Residuals/", exist_ok=True)
                        os.makedirs(f"{sharp_outFolder}Values/", exist_ok=True)
                        
                        sharpened_file = f"{sharp_outFolder}{'_'.join(highResFilename.split('.')[0].split('_')[1:5])}_{kombi}_{tile_to_process}.tif"
                        
                        runSharpi(highResFilename, lowResFilename, lowResMaskFilename, cv, movWin, regrat/100, sharpened_file, useDecisionTree = True)
                        
                        sharpList.append(sharpened_file)
            

            if len(sharpList) != 0:   
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

                    # Sensitivity
                    bioL=[False, bio_pars] 
                    C_HEIGHTL=['lai','fix']
                    T_HEIGHTL=['high','low']
                    LAND_CL=['fix','th']
                    combis = [[bi, chei, thei, lanc] for bi in bioL for chei in C_HEIGHTL for thei in T_HEIGHTL for lanc in LAND_CL]
                    bioL, C_HEIGHTL, T_HEIGHTL, LAND_CL = map(list, zip(*combis))

                    for bio, C_HEIGHT, T_HEIGHT, LAND_C in zip(bioL, C_HEIGHTL, T_HEIGHTL, LAND_CL):
        
                        # check if file already produced
                        if bio:
                            sensi_suff = f"withBio_hc{C_HEIGHT}_TMH{T_HEIGHT}_LC{LAND_C}"
                        else:
                            sensi_suff = f"withoutBio_hc{C_HEIGHT}_TMH{T_HEIGHT}_LC{LAND_C}"

                            storPath_c_f = f'{evap_outFolder}{comp}_{year}_{month}_{day}_{mvwin}_{cv}_{regrat}_{lstMask}_{s2Mask}_{sharp}_{tile}_{sensi_suff}_ET_Canopy_func.tif'
                            storPath_s_f = f'{evap_outFolder}{comp}_{year}_{month}_{day}_{mvwin}_{cv}_{regrat}_{lstMask}_{s2Mask}_{sharp}_{tile}_{sensi_suff}_ET_Soil_func.tif'
                            
                            if os.path.exists(storPath_c_f) and os.path.exists(storPath_s_f):
                                continue
                            else:
                                runEvapi(year=year, month=month, day=day, comp=comp, sharp=sharp, s2Mask=s2Mask, lstMask=lstMask, tile=tile,
                                        tempDir=trash_path, path_to_temp=temp_dump_fold, path_to_sharp=sharp_outFolder, mvwin=mvwin, cv=cv,
                                        regrat=regrat, evap_outFolder=evap_outFolder, S2path=S2_path, th_arr=th_arr, printInterim=printEvapInter,
                                        bio=bio, C_HEIGHT=C_HEIGHT, T_HEIGHT=T_HEIGHT, LAND_C=LAND_C)
                    
            
            killDates = [f"{year}_{v['month']}_{v['band']:02d}" for k, v in band_dict.items()]

            [os.remove(file) for file in getFilelist(temp_dump_fold, '.vrt') if any(kill in file for kill in killDates)]
            [os.remove(file) for file in getFilelist(temp_dump_fold, '.tif') if any(kill in file for kill in killDates)]

            if path_to_dem:
                [os.remove(file) for file in getFilelist(f"{temp_dump_fold}bio/", '.tif') if any(kill in file for kill in killDates)]
                [os.remove(file) for file in getFilelist(f"{temp_dump_fold}sun/", '.tif') if any(kill in file for kill in killDates)]

            [os.remove(file) for file in getFilelist(sharp_outFolder, '.tif', deep=True) if any(kill in file for kill in killDates)]

        
            # export the processed dates
            df = pd.DataFrame({
                'compdates': [f"{compDate}"],
                })
            df.to_csv(compdate_path, index=False)

