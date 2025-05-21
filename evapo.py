import datetime
from osgeo import gdal, osr
gtiff_driver = gdal.GetDriverByName('GTiff')

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
            date = datetime.date(year, month, day)

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

def warp_to_template(source_ds, reference_path, outPath, mask_path, resampling='bilinear', outType=gdal.GDT_UInt16):
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
        return warped_ds
