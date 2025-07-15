import os
import random
import time
import math

root_path = r'TrajData/Geolife Trajectories 1.3/Data/'
out_path = r'TrajData/Geolife_out/'
file_list = []#放所有文件的路径
dir_list = []#放所有文件夹的路径

def get_file_path(root_path,file_list,dir_list):
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        dir_file_path = os.path.join(root_path,dir_file)
        if os.path.isdir(dir_file_path):
            dir_list.append(dir_file_path)
            get_file_path(dir_file_path,file_list,dir_list)
        else:
            file_list.append(dir_file_path)
 
get_file_path(root_path, file_list, dir_list)

random.shuffle(file_list)#把所有轨迹文件顺序打乱

write_name = 0

def lonlat2meters(lon, lat):#经纬度转米，处理地图投影的转换。
    semimajoraxis = 6378137.0
    east = lon * 0.017453292519943295
    north = lat * 0.017453292519943295
    t = math.sin(north)
    return semimajoraxis * east, 3189068.5 * math.log((1 + t) / (1 - t))

for fl in file_list:
    if write_name % 100 == 0:#每一百个打印提示信息
        print('preprocessing ', write_name)
    f = open(fl)
    fw = open(out_path + str(write_name), 'w')
    c = 0
    line_count = 0
    for line in f:#每行记录一个轨迹点
        if c < 6: #跳过前六行
            c = c + 1
            continue
        temp = line.strip().split(',')
        if len(temp) < 7:
            continue
        #每个轨迹点由经纬度和时间戳表示
        #时间戳是先把年月日时分秒转化成时间数据结构，然后转换成自纪元以来秒数，取整化成字符串
        lon_meter, lat_meter = lonlat2meters(float(temp[1]), float(temp[0]))
        fw.write(str(lat_meter)+' '+str(lon_meter)+' '+str(int(time.mktime(time.strptime(temp[5]+' '+temp[6],'%Y-%m-%d %H:%M:%S'))))+'\n')
        line_count = line_count + 1
    f.close()
    fw.close()
    if line_count <= 30:#轨迹点数量少于30的舍去
        os.remove(out_path + str(write_name))
        write_name = write_name - 1
    write_name = write_name + 1    