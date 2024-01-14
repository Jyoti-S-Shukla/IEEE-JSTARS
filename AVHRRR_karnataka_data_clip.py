from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler


path = r"F:\Jyoti Shukla -MS\AVHRR global data (1981-2022)"
shapefile = r"F:\Jyoti Shukla -MS\Shapefiles for ROI\State.shp"
output_path = r"F:\Jyoti Shukla -MS\AVHRR Karnataka data (1981-2022)"
dir_list = os.listdir(path)
# adding more missing data
target_list = dir_list[995:1099]

for file in dir_list:
    if file.endswith(".tif"):        
        image = os.path.join(path,file)
        ds = gdal.Open(image)
        array = ds.GetRasterBand(1).ReadAsArray()
        #plt.imshow(array)
        # print(ds.GetGeoTransform())
        # print("==================space===============")
        # print(ds.GetProjection())
        #dsRprj = gdal.Warp("projected_1.tif",ds,dstSRS=("EPSG:4326"))
        outfile = os.path.join(output_path,file)
        dsClip = gdal.Warp(outfile,ds,cutlineDSName=(shapefile),cropToCutline=(True),dstNodata=(-3.4e+38))
        
        
#..........................array to image .tif format....................................................#
#========================================================================================================#
def array_to_image(array,dtype,outfile):  
    image=gdal.Open(path)     
    trans = image.GetGeoTransform()
    proj = image.GetProjection()
    #nodata= image.GetRasterBand(1).GetNoDataValue()
    #out ="testimage.tif"
    outdriver = gdal.GetDriverByName("GTIFF")
    outdata = outdriver.Create(str(outfile),image.RasterXSize, image.RasterYSize, 1,gdal.GDT_Float32)
    outdata.GetRasterBand(1).WriteArray(array)
    #outdata.GetRasterBand(1).SetNoDataValue(nodata)
    outdata.SetGeoTransform(trans)
    outdata.SetProjection(proj)
    outdata=None        
        
# ..................converting image values to .txt values...............................................#
#========================================================================================================#

image_dir = os.listdir(output_path)
AVHRR_data_1981_2022 =[]

for image_name in image_dir:
    if image_name.endswith(".tif"):
        im = gdal.Open(os.path.join(output_path,image_name))
        img_array = im.GetRasterBand(1).ReadAsArray()
        AVHRR_data_1981_2022.append(img_array)
    

VHI_image_train = AVHRR_data_1981_2022[0:2070]
VHI_image_test = AVHRR_data_1981_2022[2070:2123]

#saving and reading as txt file

vhi_train_file = open(r"F:\Jyoti Shukla -MS\karnataka dataset/Train_total_AVHRR_data_1981_2022.txt","w")
for array in VHI_image_train:
  np.savetxt(vhi_train_file,array)
vhi_train_file.close()

vhi_test_file = open(r"F:\Jyoti Shukla -MS\karnataka dataset/Test_total_AVHRR_data_1981_2022.txt","w")
for array in VHI_image_test:
  np.savetxt(vhi_test_file,array)
vhi_test_file.close()

#.................loading data..........................................................................#
VHI_train = np.loadtxt(r"F:\Jyoti Shukla -MS\karnataka dataset/Train_total_AVHRR_data_1981_2022.txt").reshape(2070,190,124)
VHI_test = np.loadtxt(r"F:\Jyoti Shukla -MS\karnataka dataset/Test_total_AVHRR_data_1981_2022.txt").reshape(53,190,124)
VHI_train, VHI_test = VHI_train.astype("float32"), VHI_test.astype("float32")
np.max(VHI_train), np.min(VHI_train), np.max(VHI_test), np.min(VHI_test)

#.....................................saving the padded array and labels................................#
#=======================================================================================================#
VHI_train_padded_file = open(r"F:\Jyoti Shukla -MS\karnataka dataset/Padded_Train_total_AVHRR_data_1981_2022.txt","w")
for i in range(len(VHI_train)):
  VHI_train[i] = np.where(VHI_train[i]<0,0,VHI_train[i]) #removing nodata
  #VHI_train[i+1] = np.where(VHI_train[i+1]<0,0,VHI_train[i+1])
  pad = np.pad(VHI_train[i],((2,0),(0,4))) # padding to get divisible by 2 shape (190,128)
  #pad_label = np.pad(VHI_train[i+1],((2,0),(0,4)))
  np.savetxt(VHI_train_padded_file,pad)
VHI_train_padded_file.close()

VHI_test_padded_file = open(r"F:\Jyoti Shukla -MS\karnataka dataset/Padded_Test_total_AVHRR_data_1981_2022.txt","w")
for i in range(len(VHI_test)):
  VHI_test[i] = np.where(VHI_test[i]<0,0,VHI_test[i]) #removing nodata
  #VHI_train[i+1] = np.where(VHI_train[i+1]<0,0,VHI_train[i+1])
  pad = np.pad(VHI_train[i],((2,0),(0,4))) # padding to get divisible by 2 shape (190,128)
  #pad_label = np.pad(VHI_train[i+1],((2,0),(0,4)))
  np.savetxt(VHI_test_padded_file,pad)
VHI_test_padded_file.close()

#....................................normalizing data within 0 and 1.....................................#
#========================================================================================================#

train_array = []
train_labels = []
scales = []
for i in range(len(VHI_train)-1):
  VHI_train[i] = np.where(VHI_train[i]<0,0,VHI_train[i]) #removing nodata
  VHI_train[i+1] = np.where(VHI_train[i+1]<0,0,VHI_train[i+1])
  pad = np.pad(VHI_train[i],((2,0),(0,4))) # padding to get divisible by 2 shape (190,128)
  pad_label = np.pad(VHI_train[i+1],((2,0),(0,4)))
  #scaling in 0 to 1 range
  scaler = MinMaxScaler()
  scaled = scaler.fit_transform(pad)
  scales.append(scaled)
  scaled_label = scaler.fit_transform(pad_label)
  train_array.append((scaled.reshape(192,128,1))) # giving the next image as label for the current image
  train_labels.append((scaled_label.reshape(192,128,1)))
len(train_labels), train_array[1].shape
plt.imshow(train_array[0].reshape(192,128))


unscaled_train = []
for i in range(len(VHI_train)-1):
  VHI_train[i] = np.where(VHI_train[i]<0,0,VHI_train[i]) #removing nodata
  VHI_train[i+1] = np.where(VHI_train[i+1]<0,0,VHI_train[i+1])
  pad = np.pad(VHI_train[i],((2,0),(0,4))) # padding to get divisible by 2 shape (190,128)
  pad_label = np.pad(VHI_train[i+1],((2,0),(0,4)))
  #scaling in 0 to 1 range
  scaler = MinMaxScaler()
  scaled = scaler.fit_transform(pad)
  unscaled = scaler.inverse_transform(train_array[i].reshape(192,128))
  unscaled_train.append(unscaled)
  
test_array = []
test_labels = []
for i in range(len(VHI_test)-1):
  VHI_test[i] = np.where(VHI_test[i]<0,0,VHI_test[i]) #removing nodata
  VHI_test[i+1] = np.where(VHI_test[i+1]<0,0,VHI_test[i+1])
  pad = np.pad(VHI_test[i],((2,0),(0,4))) # padding to get divisible by 2 shape (190,128)
  pad_label = np.pad(VHI_test[i+1],((2,0),(0,4)))
  #scaling in 0 to 1 range
  scaler = MinMaxScaler()
  scaled = scaler.fit_transform(pad)
  #scales.append(scaled)
  scaled_label = scaler.fit_transform(pad_label)
  test_array.append(scaled.reshape(192,128,1)) # giving the next image as label for the current image
  test_labels.append(scaled_label.reshape(192,128,1))
len(test_labels)

#............................................storing as paired array.....................................#
#========================================================================================================#

x = train_array
y = train_labels
np.savez("F:\Jyoti Shukla -MS\karnataka dataset/paired_arrays_train_total_AVHRR_data.npz",x = x , y = y)

x = test_array
y = test_labels
np.savez("F:\Jyoti Shukla -MS\karnataka dataset/paired_arrays_test_total_AVHRR_data.npz",x = x , y = y)

np.max(test_array), np.max(train_array), np.min(test_array), np.min(train_array)
