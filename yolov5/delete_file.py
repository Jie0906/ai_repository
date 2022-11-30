import os
import datetime
import datetime 
import logging
import time
import psycopg2


class DeleteFile(object):
    def __init__(self,path):
        self.path = path

    def delete(self):
        """
        删除文件
        :param path: 文件路径
        :return: bool
        """
        file_list = [self.path]  # 文件夹列表
        # 获取当前时间
        today = datetime.datetime.now()
        # 计算偏移量,前3天
        offset =  datetime.timedelta(hours=-1)
        # 获取想要的日期的时间,即前3天时间
        re_date = (today + offset)
        # 前3天时间转换为时间戳
        re_date_unix = time.mktime(re_date.timetuple())


        try:
            while file_list:  # 判断列表是否为空
                path = file_list.pop()  # 删除列表最后一个元素，并返回给path l = ['E:\python_script\day26']
                for item in os.listdir(path):  # 遍历列表,path = 'E:\python_script\day26'
                    path2 = os.path.join(path, item)  # 组合绝对路径 path2 = 'E:\python_script\day26\test'
                    if os.path.isfile(path2):  # 判断绝对路径是否为文件
                        # 比较时间戳,文件修改时间小于等于3天前
                        if os.path.getmtime(path2) <= re_date_unix:
                            os.remove(path2)

                    else:
                        if not os.listdir(path2):  # 判断目录是否为空
                            # 若目录为空，则删除，并递归到上一级目录，如若也为空，则删除，依此类推
                            os.removedirs(path2)
                
                        else:
                            # 为文件夹时,添加到列表中。再次循环。l = ['E:\python_script\day26\test']
                            file_list.append(path2)

            return True
        except Exception as e:
            print(e)
            return False


    def  del_database():
            host = '220.133.51.96'
            port = '5433'
            dbname = 'agv'
            user = 'agvai'
            password = 'agvai1qaz'
            conn_string = 'host = {} port = {} user = {} dbname = {} password = {}'.format(host, port, user, dbname, password)
            conn = psycopg2.connect(conn_string)
            conn.autocommit = True
            print("DB connected sucess!")
            cursor = conn.cursor()
            cursor.execute("DELETE FROM agv WHERE create_at < (now() - '1 hour'::interval);")




# while True:
#     try:
#         ret = DeleteFile('C:/Users/user/Desktop/yolo_box/yolov5/runs/detect').delete()  # 当前目录
#         print("Delete result: "+str(ret))
#         print(datetime.datetime.now())
#         time.sleep(10)

#     except Exception as e:
#         print(e)

ret = DeleteFile('C:/Users/user/Desktop/yolo_box/yolov5/runs/detect').delete()
DeleteFile.del_database()
