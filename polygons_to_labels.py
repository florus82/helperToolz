import cv2
import glob
import geopandas as gpd
import numpy as np
from osgeo import gdal, ogr, osr
import matplotlib.pyplot as plt
import os

# 00_polygons_to_lines.py
def polygons_to_lines(path_to_polygon, path_to_lines_out, categories=None):
    
    if not os.path.exists(path_to_lines_out):
        # Load the GeoParquet file
        gdf = gpd.read_parquet(path_to_polygon)

        # Ensure the geometries are polygons
        if not all(gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])):
            raise ValueError('The file must contain Polygon or MultiPolygon geometries.')

        # Exclude specified categories if provided
        if categories is not None:
            initial_count = len(gdf)
            gdf = gdf[~gdf['EC_hcat_n'].isin(categories)]
            filtered_count = initial_count - len(gdf)
            print(f"Filtered out {filtered_count} rows from {initial_count} based on categories")

        # Convert polygons to lines
        gdf['geometry'] = gdf.geometry.boundary
        #export
        gdf.to_file(path_to_lines_out, driver='GPKG')  

        print("Conversion complete: Polygons converted to lines.")
    else:
        print(f'Polylines for polygon {path_to_polygon} already exists!!!')


# 01_rasterize_line_feats
def rasterize_lines(path_to_lines, path_to_extent_raster, path_to_rasterlines_out, all_touch=True):

    ##### open field vector file
    field = ogr.Open(path_to_lines)
    field_lyr = field.GetLayer(0)

    if not os.path.exists(path_to_rasterlines_out):
        ds = gdal.Open(path_to_extent_raster)
        target_ds = gdal.GetDriverByName('GTiff').Create(path_to_rasterlines_out, ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Byte)
        target_ds.SetGeoTransform(ds.GetGeoTransform())
        target_ds.SetProjection(ds.GetProjection())

        if all_touch:
             opti = ["ALL_TOUCHED=TRUE"]
        else:
             opti = ["ALL_TOUCHED=FALSE"]
        gdal.RasterizeLayer(target_ds, [1], field_lyr, burn_values=[1], options = opti)
        target_ds = None
    else:
        print(f'Rasterized lines for {path_to_lines} already exists!!!')


# 02_multitask_labels
def make_multitask_labels(path_to_rasterlines, path_to_mtsk_out):

    edge = cv2.imread(path_to_rasterlines, cv2.IMREAD_GRAYSCALE)
    crop = get_crop(edge)
    dist = get_distance(crop)
    edge = cv2.dilate(edge, np.ones((2,2), np.uint8), 1)

    label = np.stack([crop, edge, dist])
    mem_ds = create_mem_ds(path_to_rasterlines, 3)

    # write outputs to bands
    for b in range(3):
        mem_ds.GetRasterBand(b+1).WriteArray(label[b,:,:])

    # create physical copy of ds
    copy_mem_ds(path_to_mtsk_out, mem_ds)

# 03 create a crop mask
def make_crop_mask(path_to_polygon, path_to_rasterized_lines, path_to_extent_raster, path_to_mask_out, all_touch=True):
    #### open field vector file
    field_gpd = gpd.read_parquet(path_to_polygon)

    # convert field_gpd to vector layer for rasterization
    ogr_ds = ogr.GetDriverByName('Memory').CreateDataSource('')
    srs = ogr.osr.SpatialReference()
    srs.ImportFromWkt(field_gpd.crs.to_wkt())
    field_lyr = ogr_ds.CreateLayer("field_layer", srs, ogr.wkbPolygon)

    # Convert geometries
    layer_defn = field_lyr.GetLayerDefn()
    for _, row in field_gpd.iterrows():
        feature = ogr.Feature(layer_defn)

        # Validate geometry
        if row.geometry is None or row.geometry.is_empty:
            print("Skipping empty geometry.")
            continue

        # Convert to OGR geometry
        geom = ogr.CreateGeometryFromWkb(row.geometry.wkb)
        if geom is None:
            print("Invalid geometry encountered, skipping.")
            continue

        feature.SetGeometry(geom)
        field_lyr.CreateFeature(feature)
        feature = None  # Free memory

    if not os.path.exists(path_to_mask_out):
        ds = gdal.Open(path_to_extent_raster)
        target_ds = gdal.GetDriverByName('GTiff').Create(path_to_mask_out, ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Byte)
        target_ds.SetGeoTransform(ds.GetGeoTransform())
        target_ds.SetProjection(ds.GetProjection())

        # this choice appears to have no effect when rasterizing the polygons
        if all_touch:
            opti = ["ALL_TOUCHED=TRUE"]
        else:
            opti = ["ALL_TOUCHED=FALSE"]
        gdal.RasterizeLayer(target_ds, [1], field_lyr, burn_values=[1], options = opti)
        target_ds = None
    else:
        print(f'Mask for {path_to_polygon} already exists!!!')
    # mask the output with rasterized lines to clean up

    path_linecrop_out = path_to_mask_out.split('.')[0] + '_lines_touch_' + path_to_rasterized_lines.split('_')[-1].split('.')[0] + '_linecrop.tif'
    if not os.path.exists(path_linecrop_out):
        mask_ds = gdal.Open(path_to_mask_out)
        mask = mask_ds.GetRasterBand(1).ReadAsArray()
        lines_ds  = gdal.Open(path_to_rasterized_lines)
        lines = lines_ds.GetRasterBand(1).ReadAsArray()
        
        mask[np.where(lines == 1)] = 0

        target_ds = gdal.GetDriverByName('GTiff').Create(path_linecrop_out, mask_ds.RasterXSize, mask_ds.RasterYSize, 1, gdal.GDT_Byte)
        target_ds.SetGeoTransform(mask_ds.GetGeoTransform())
        target_ds.SetProjection(mask_ds.GetProjection())
        target_ds.GetRasterBand(1).WriteArray(mask)
        del target_ds
    else:
        print(f'Mask for {path_to_polygon} in combination with {path_to_rasterized_lines} already exists!!!')

###########################
####### helper functions

# create dataset in memory using geotransform specified in ref_pth
def create_mem_ds(ref_pth, n_bands):
        drvMemR = gdal.GetDriverByName('MEM')
        ds = gdal.Open(ref_pth)
        mem_ds = drvMemR.Create('', ds.RasterXSize, ds.RasterYSize, n_bands, gdal.GDT_Float32)
        mem_ds.SetGeoTransform(ds.GetGeoTransform())
        mem_ds.SetProjection(ds.GetProjection())
        return mem_ds

# create copy
def copy_mem_ds(pth, mem_ds):
        copy_ds = gdal.GetDriverByName("GTiff").CreateCopy(pth, mem_ds, 0, options=['COMPRESS=LZW'])
        copy_ds = None

######################
# multi-taks labels from boundaries
def get_boundary(label, kernel_size = (2,2)):
    tlabel = label.astype(np.uint8)
    temp = cv2.Canny(tlabel,0,1)
    tlabel = cv2.dilate(
        temp,
        cv2.getStructuringElement(
            cv2.MORPH_CROSS,
            kernel_size),
        iterations = 1)
    tlabel = tlabel.astype(np.float32)
    tlabel /= 255.
    return tlabel

def get_distance(label):
    tlabel = label.astype(np.uint8)
    dist = cv2.distanceTransform(tlabel,
                                 cv2.DIST_L2,
                                 0)

    # get unique objects
    output = cv2.connectedComponentsWithStats(label, 4, cv2.CV_32S)
    num_objects = output[0]
    labels = output[1]

    # min/max normalize dist for each object
    for l in range(num_objects):
        dist[labels==l] = (dist[labels==l]) / (dist[labels==l].max())

    return dist

def get_crop(image, kernel_size = (3,3)):

    im_floodfill = image.copy()
    h, w = image.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # floodfill
    cv2.floodFill(im_floodfill, mask, (0,0), 1);

    # invert
    im_floodfill = cv2.bitwise_not(im_floodfill)

    # kernel size
    kernel = np.ones(kernel_size, np.uint8)

    # erode & dilate
    img_erosion = cv2.erode(im_floodfill, kernel, iterations=1)
    return cv2.dilate(img_erosion, kernel, iterations=1) - 254

