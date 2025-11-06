import sys
sys.path.append('/home/potzschf/repos/')
from helperToolz.helpsters import *
from datetime import datetime, timedelta
from PIL import Image
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.enums import Resampling

tiff_file_input = '/data/Aldhani/eoagritwin/fields/output/predictions/FORCE/BRANDENBURG/masked_predictions/256_20_chips_GSA-DE_BRB-2019_cropMask_lines_touch_true_lines_touch_true_linecrop_prediction_extent.tif'
outPath = '/data/Aldhani/eoagritwin/fields/output/predictions/FORCE/BRANDENBURG/masked_predictions/make_it_small/2019_cropMask_lines_touch_true_lines_touch_true_linecrop_prediction_extent'
output_kml_path = f'{outPath}.kml'


color_ramp = [
    (0, "#000004"),
    (0.0196078, "#02020b"),
    (0.0392157, "#050416"),
    (0.0588235, "#090720"),
    (0.0784314, "#0e0b2b"),
    (0.09803920000000001, "#140e36"),
    (0.117647, "#1a1036"),
    (0.13725499999999999, "#21114e"),
    (0.156863, "#29115a"),
    (0.17647099999999999, "#311165"),
    (0.196078, "#390f6e"),
    (0.21568599999999999, "#420f75"),
    (0.235294, "#4a1079"),
    (0.25490200000000002, "#52137c"),
    (0.27450999999999998, "#5e197e"),
    (0.29411799999999999, "#661c80"),
    (0.31372499999999998, "#6a1d81"),
    (0.33333299999999999, "#6e1f81"),
    (0.352941, "#792281"),
    (0.37254900000000002, "#812381"),
    (0.39215699999999998, "#892580"),
    (0.41176499999999999, "#912780"),
    (0.43137300000000001, "#992980"),
    (0.45097999999999999, "#a22f7e"),
    (0.47058800000000001, "#aa317d"),
    (0.49019600000000002, "#b2377c"),
    (0.50980400000000003, "#ba3d79"),
    (0.52941199999999999, "#c24376"),
    (0.54901999999999995, "#ca4b72"),
    (0.56862699999999999, "#d3536c"),
    (0.58823499999999995, "#db5c67"),
    (0.60784300000000002, "#e26761"),
    (0.62745099999999998, "#e9735a"),
    (0.64705900000000005, "#ef7e55"),
    (0.66666700000000001, "#f38b51"),
    (0.68627499999999997, "#f7984e"),
    (0.70588200000000001, "#fba54c"),
    (0.72548999999999997, "#fdb04c"),
    (0.74509800000000004, "#feb94f"),
    (0.764706, "#fdc259"),
    (0.78431399999999996, "#fdcb65"),
    (0.80392200000000003, "#fdd371"),
    (0.82352899999999996, "#fddb7e"),
    (0.84313700000000003, "#fee28b"),
    (0.86274499999999998, "#ffe998"),
    (0.88235300000000005, "#ffefa5"),
    (0.90196100000000001, "#fff5b3"),
    (0.92156899999999997, "#fffbc0"),
    (0.94117600000000001, "#fffecd"),
    (0.96078399999999997, "#fffdda"),
    (0.98039200000000004, "#fffbe6"),
    (1, "#fff9ef")
]


# Quick funktion to convert to split hex into RGBA colors, as needed later
def hex_to_rgba(hex_color, alpha=255):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4)) + (alpha,)

def map_value_to_color(value):
    for upper_bound, hex_color in color_ramp:
        if value <= upper_bound:
            return hex_to_rgba(hex_color)
    return (0, 0, 0, 0)

ds = gdal.Open(tiff_file_input, 0)
bands = ds.RasterCount
png_files = []
for i in range(bands):
    # band = ds.GetRasterBand(i + 1).ReadAsArray()

    # # Create an empty RGBA array
    # rgba_image = np.zeros((band.shape[0], band.shape[1], 4), dtype=np.uint8)

    # # Flatten band1 and map each pixel's value to color
    # flat_band = band.flatten()

    # print('create array')
    # # Create an array to hold RGBA values
    # colors = np.array([map_value_to_color(val) for val in flat_band], dtype=np.uint8)
    # print('start reshape')
    # # Reshape colors to image shape
    # rgba_image = colors.reshape((band.shape[0], band.shape[1], 4))
    # print('start conversion')
    # # Convert numpy RGBA array to PIL Image and save
    # im = Image.fromarray(rgba_image, mode="RGBA")
    # print('write away')
    # im.save(f'{outPath}_band_{i+1}.png', "PNG")
    png_files.append(f'{outPath}_band_{i+1}.png')

# get bounding boxes
bboxes = []
with rasterio.open(tiff_file_input) as dataset:
    bounds = dataset.bounds
    bboxes.append(bounds)

from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString


##### insert time dummy
times = []
# Monday
first_day_of_year= datetime.fromisocalendar(2019, 1, 1)
# Sunday
last_day_of_year = first_day_of_year + timedelta(days=365)

# 1 TIF = 1 calendar week in my case
start_time = first_day_of_year.strftime("%Y-%m-%dT00:00:00Z")
end_time = last_day_of_year.strftime("%Y-%m-%dT23:59:59Z")

times.append((start_time, end_time))


kml_files = []

kml_start = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://earth.google.com/kml/2.0">
<Document>"""

kml_end = """</Document>
</kml>"""

filled_kml = ""

for png_filename, (west, south, east, north), (begin, end) in zip(png_files, bboxes, times):    
    kml_template = """
 <Folder>
    <name>Raster visibility</name>
    <TimeSpan><begin>{begin}</begin><end>{end}</end></TimeSpan>
    <Folder>
      <name>Raster</name>
      <GroundOverlay>
          <name>Raster data</name>
          <LatLonBox>
            <north>{north}</north>
            <south>{south}</south>
            <west>{west}</west>
            <east>{east}</east>
          </LatLonBox>
          <Icon>
            <href>{href}</href>
          </Icon>
      </GroundOverlay>
    </Folder>
  </Folder>"""
    filled_kml += kml_template.format(
        begin=begin,
        end=end,
        north=north,
        south=south,
        west=west,
        east=east,
        href=os.path.basename(png_filename)
    )

filled_kml = f"{kml_start}{filled_kml}{kml_end}"

with open(output_kml_path, 'w') as f:
    f.write(filled_kml)
print(f"KML file {output_kml_path} generated successfully")


# generate the kmz file
import zipfile

base_name = os.path.splitext(os.path.basename(output_kml_path))[0]
kmz_path = os.path.join(os.path.dirname(output_kml_path), base_name + '.kmz')

with zipfile.ZipFile(kmz_path, 'w', zipfile.ZIP_DEFLATED) as kmz:
    kmz.write(output_kml_path, os.path.basename(output_kml_path))
    for png_path in png_files:    
        kmz.write(png_path, os.path.basename(png_path))

print(f"KMZ created at: {kmz_path}")