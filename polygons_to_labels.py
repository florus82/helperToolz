import cv2
import glob
import geopandas as gpd
import pandas as pd
import numpy as np
from osgeo import gdal, ogr, osr
import matplotlib.pyplot as plt
import os
import rasterio
from scipy import ndimage

EXCLUDE_LIST = ['',
 '(Beta-)Rübensamenvermehrung',
 'Alle anderen Flächen (keine LF)',
 'Baumschulen, nicht für Beerenobst',
 'Bestockte Rebfläche',
 'Erosionsschutzstreifen',
 'Forstflächen (Waldbodenflächen)',
 'Gewässerschutzstreifen',
 'Grassamenvermehrung',
 'Haus- und Nutzgärten',
 'KUP lt. Direktzahlungendurchführungsverordnung',
 'KUP lt. Direktzahlungendurchführungsverordnung (keine ÖVF)',
 'KUP lt. GAPDZV',
 'N. LNF, n. Art. 32(2b(i)) der VO(EG) Nr.1307/2013 beihilfef. Fl.',
 'Nicht landwirt. Fl. In der Verfügungsgewalt des Antragstellers, die gem. § 15 (1) DirektZahlDurchfG als umweltsensibles Dauergrünland bestimmt worden sind',
 'Nicht landwirt. Fl. infolge Genehmigung DGL Umwandlung',
 'Nicht landwirtschaftliche, aber nach Art. 32(2b (i)) der VO (EG) Nr. 1307/2013 beihilfefähige Fläche',
 'Nicht landwirtschaftliche, aber nach Art. 32(2b (i)) der VO (EG) Nr. 1307/2013 beihilfefähige Fläche (Naturschutzflächen, die 2008 noch beihilfefähig waren)',
 'Nicht landwirtschaftliche, aber nach §11 (1) Nr.3 Bst. a) bb) der GAPDZV förderfähige Fläche (Infolge Anwendung der Wasserrahmenrichtlinie)',
 'Nicht landwirtschaftliche, aber §11 (1) Nr.3 Bst. c) der GAPDZV förderfähige Fläche (Aufforstungsverpflichtung nach VO 1257/1999 oder VO (EG) Nr. 1698/2005 oder VO 1305/2013 oder VO 2021/2115 oder bei Eingehung damit in Einklang stehender öffentlich',
 'None',
 'Pufferstreifen ÖVF DGL',
 'Schonstreifen',
 'Streifen am Waldrand (ohne Produktion) ÖVF',
 'Ufervegetation ÖVF',
 'Unbefestigte Mieten-, Stroh-, Futter und Dunglagerplätze auf AL',
 'Unbefestigte Mieten-, Stroh-, Futter und Dunglagerplätze auf DGL',
 'Unbestockte Rebfläche',
 'Vorübergehende, unbefestigte Mieten-, Stroh-, Futter und Dunglagerplätze auf AL',
 'Vorübergehende, unbefestigte Mieten-, Stroh-, Futter und Dunglagerplätze auf DGL',
 'Weihnachtsbäume',
 'Wildäsungsfläche',
 'afforestation_reforestation',
 'alle anderen Flächen (keine LF)',
 'aufgeforstete Dauergrünlandflächen, weder nach 1257/99 oder VO (EG) Nr. 1698/2005  1305/2013oder VO (EU) Nr.1305/2013',
 'aufgeforstete Flächen (VO1257/1999, 1698/2005, 1305/2013)',
 'greenhouse_foil_film',
 'nach VO 1257/1999 oder VO (EG) Nr. 1698/2005 oder VO 1305/2013 aufgeforstete Flächen',
 'not_known_and_other',
 'nurseries_nursery',
 'tree_wood_forest',
 'unmaintained',
 'vorübergehend unbefestigte Mieten-, Stroh-, Futter oder Dunglagerplätze auf DGL',
 'vorübergehende, unbefestigte Mieten-, Stroh-, Futter oder Dunglagerplätze auf AL']


# 00_polygons_to_lines.py
def polygons_to_lines(path_to_polygon, path_to_lines_out, categories=None, category_col=None):
    
    if not os.path.exists(path_to_lines_out):
        # Load polygons
        gdf = gpd.read_file(path_to_polygon)
        
        # Filter categories if provided
        if categories is not None and category_col is not None:
            gdf = gdf[~gdf[category_col].isin(categories)]
        
        # Keep only valid geometries
        gdf = gdf[gdf.is_valid]
        
        # Drop duplicate geometries
        gdf = gdf.drop_duplicates(subset='geometry')
        
        # Convert polygons to lines
        gdf['geometry'] = gdf.geometry.boundary
        
        # Drop any existing 'fid' column to avoid UNIQUE constraint
        if 'fid' in gdf.columns:
            gdf = gdf.drop(columns=['fid'])
        
        # Reset index to ensure unique IDs
        gdf = gdf.reset_index(drop=True)
        
        # Construct layer name
        layer_name = path_to_polygon.split('/')[-1].split('.')[0]
        
        # Save to GeoPackage using Fiona backend (safer for new layers)
        gdf.to_file(path_to_lines_out, driver='GPKG', layer=layer_name)
        

        print("Conversion complete: Polygons converted to lines.")
    else:
        print(f'Polylines for polygon {path_to_polygon} already exists!!!')


# 01_rasterize_line_feats
def rasterize_lines(path_to_lines, path_to_extent_raster, path_to_rasterlines_out, all_touch=True):

    if not os.path.exists(path_to_rasterlines_out):
        ##### open field vector file
        field = ogr.Open(path_to_lines)
        field_lyr = field.GetLayer(0)

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
def make_crop_mask(path_to_polygon, path_to_extent_raster, path_to_mask_out, path_to_rasterized_lines=False,\
                   all_touch=True, categories=None, category_col=None, burn_col = False):
    
    if not os.path.exists(path_to_mask_out):
        #### open field vector file
        field_gpd = gpd.read_file(path_to_polygon)
        if burn_col== 'id_0':
            field_gpd = field_gpd.reset_index(drop=False).rename(columns={'index': 'id_0'})
        # Exclude specified categories if provided
        if categories is not None and category_col is not None:
            field_gpd = field_gpd[~field_gpd[category_col].isin(categories)]

        # Keep only valid geometries
        field_gpd = field_gpd[field_gpd.is_valid]
        
        # Drop duplicate geometries
        field_gpd = field_gpd.drop_duplicates(subset='geometry')

        # convert field_gpd to vector layer for rasterization
        ogr_ds = ogr.GetDriverByName('Memory').CreateDataSource('')
        srs = ogr.osr.SpatialReference()
        srs.ImportFromWkt(field_gpd.crs.to_wkt())
        field_lyr = ogr_ds.CreateLayer("field_layer", srs, ogr.wkbPolygon)
        
        if burn_col:
            # Create fields (attributes) in the OGR layer 
            for col_name, dtype in zip(field_gpd.columns, field_gpd.dtypes):
                if col_name == field_gpd.geometry.name:
                    continue  # skip geometry column

                if np.issubdtype(dtype, np.integer):
                    field_type = ogr.OFTInteger64
                elif np.issubdtype(dtype, np.floating):
                    field_type = ogr.OFTReal
                else:
                    field_type = ogr.OFTString

                field_defn = ogr.FieldDefn(col_name, field_type)
                field_lyr.CreateField(field_defn)
    
        # Convert geometries
        layer_defn = field_lyr.GetLayerDefn()
        for _, row in field_gpd.iterrows():
        
            # Validate geometry
            if row.geometry is None or row.geometry.is_empty:
                print("Skipping empty geometry.")
                continue

            # Convert to OGR geometry
            geom = ogr.CreateGeometryFromWkb(row.geometry.wkb)
            if geom is None:
                print("Invalid geometry encountered, skipping.")
                continue

            feature = ogr.Feature(layer_defn)
            feature.SetGeometry(geom)

            if burn_col:
                # Set attribute values
                for col_name in field_gpd.columns:
                    if col_name != field_gpd.geometry.name:
                        value = row[col_name]
                        if pd.notnull(value):
                            feature.SetField(col_name, value)
            field_lyr.CreateFeature(feature)
            feature = None  # Free memory

        if burn_col:
            export_dt = gdal.GDT_UInt32
        else:
            export_dt = gdal.GDT_Byte
        ds = gdal.Open(path_to_extent_raster)
        target_ds = gdal.GetDriverByName('GTiff').Create(path_to_mask_out, ds.RasterXSize, ds.RasterYSize, 1, export_dt)
        target_ds.SetGeoTransform(ds.GetGeoTransform())
        target_ds.SetProjection(ds.GetProjection())
        band = target_ds.GetRasterBand(1)
        band.SetNoDataValue(0)
        band.Fill(0)

        # this choice appears to have no effect when rasterizing the polygons
        if all_touch:
            opti = ["ALL_TOUCHED=TRUE"]
        else:
            opti = ["ALL_TOUCHED=FALSE"]

        if burn_col:
            opti.append(f"ATTRIBUTE={burn_col}")
            gdal.RasterizeLayer(target_ds, [1], field_lyr, options = opti)
        else:
            gdal.RasterizeLayer(target_ds, [1], field_lyr, burn_values=[1], options = opti)
        target_ds = None
    else:
        print(f'Mask for {path_to_polygon} already exists!!!')

    # mask the output with rasterized lines to clean up
    if path_to_rasterized_lines:
        path_linecrop_out = path_to_mask_out.split('.')[0] + '_linecrop.tif' # + '_lines_touch_' + path_to_rasterized_lines.split('_')[-1].split('.')[0]
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
    else:
        pass


def get_fieldborder_distance_raster(path_to_object, outPath):
    """Create a distance raster from a binary raster (distance to fieldborder which has value 1)
        The distance values are cropped to the inside of fields
    Args:
        path_to_object (str): path to binary raster
        outPath (str): path to output.tif on disc where distance raster will be stored
    """

    with rasterio.open(path_to_object) as src:
        border = src.read(1)
        profile = src.profile

    # Invert border: 1 inside polygon, 0 on border
    polygon_mask = border

    # Compute Euclidean distance to border (border pixels = 0)
    distance = ndimage.distance_transform_edt(polygon_mask)

    # Mask outside polygons (if needed, e.g., if border raster has extra 0s)
    distance[polygon_mask == 0] = 0

    # Normalize distances inside polygons
    max_dist = distance.max()
    distance_scaled = np.zeros_like(distance, dtype=np.float32)
    if max_dist > 0:
        distance_scaled[distance > 0] = distance[distance > 0] / max_dist

    # Save raster
    profile.update(dtype=rasterio.float32, count=1)
    with rasterio.open(outPath, "w", **profile) as dst:
        dst.write(distance_scaled, 1)

    # from scipy.ndimage import distance_transform_edt, maximum
    # with rasterio.open(path_to_object) as src:
    #     arr = src.read(1)
    #     profile = src.profile

    # # Initialize normalized distance array
    # dist_norm = np.zeros_like(arr, dtype=np.float32)

    # # Mask of all field pixels
    # field_mask = arr != 0

    # # Compute distance to nearest background pixel (0)
    # # distance_transform_edt calculates distance to 0 pixels
    # dist_all = distance_transform_edt(field_mask)

    # # Assign distance only to field pixels
    # dist_norm[field_mask] = dist_all[field_mask]

    # # Normalize per polygon using scipy.ndimage.maximum
    # unique_ids = np.unique(arr)
    # unique_ids = unique_ids[unique_ids != 0]  # skip background

    # # Compute max distance per polygon
    # max_dist_per_id = maximum(dist_norm, labels=arr, index=unique_ids)

    # # Create a mapping from polygon ID -> max distance
    # max_dict = dict(zip(unique_ids, max_dist_per_id))

    # # Normalize distances per polygon
    # vectorized_norm = np.zeros_like(dist_norm, dtype=np.float32)
    # for fid, max_val in max_dict.items():
    #     if max_val > 0:
    #         vectorized_norm[arr == fid] = dist_norm[arr == fid] / max_val

    # # Save normalized distance raster
    # with rasterio.open(outPath, 'w', **profile) as dst:
    #     dst.write(vectorized_norm, 1)


def get_distance_raster(path_to_object, outPath):
    """Create a distance raster from a binary raster (distance to value 1)

    Args:
        path_to_object (str): path to binary raster
        outPath (str): path to output.tif on disc where distance raster will be stored
    """
    if not os.path.exists(outPath):
        # Open the input
        src_ds = gdal.Open(path_to_object)
        # Create output raster
        driver = gdal.GetDriverByName('GTiff')
        dst_ds = driver.Create(
            outPath,
            src_ds.RasterXSize,
            src_ds.RasterYSize,
            1,
            gdal.GDT_Float32
        )
        dst_ds.SetGeoTransform(src_ds.GetGeoTransform())
        dst_ds.SetProjection(src_ds.GetProjection())

        # Compute distance raster
        gdal.ComputeProximity(
            src_ds.GetRasterBand(1),
            dst_ds.GetRasterBand(1),
            options=["VALUES=1", "DISTUNITS=GEO"]
        )

        dst_ds = None

        print(f'Distance raster for {path_to_object} created')
    else:
        print(f'distance raster for {path_to_object} already exists!!!')

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

