from distutils.log import error
from msilib.schema import Error
from turtle import title
import psycopg2
import json
import os
from threading import Timer
from datetime import datetime
import time

class DbHandler(object):
    def __init__(self, cursor) :
        self.cursor = cursor

    def create_table(self):
        commands = (
            """
            CREATE TABLE IF NOT EXISTS agv (
                id SERIAL PRIMARY KEY,
                camera_id VARCHAR(255) NOT NULL,
                img_filename VARCHAR(255) NOT NULL,
                p1 VARCHAR(255) NOT NULL,
                p2 VARCHAR(255) NOT NULL,
                p3 VARCHAR(255) NOT NULL,
                p4 VARCHAR(255) NOT NULL,
                p0 VARCHAR(255) NOT NULL,
                other_object BOOLEAN NOT NULL,
                image_time VARCHAR(255) NOT NULL,
                detect_result BOOLEAN NOT NULL,
                object_type VARCHAR(255) NOT NULL,
                confidence INT NOT NULL,
                create_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT current_timestamp 

            )"""
            
            )

        # CREATE TABLE agv (
        #         id SERIAL PRIMARY KEY,
        #         camera_id VARCHAR(255) NOT NULL,
        #         img_filename VARCHAR(255) NOT NULL,
        #         area_id VARCHAR(255) NOT NULL,
        #         p1 VARCHAR(255) NOT NULL,
        #         p2 VARCHAR(255) NOT NULL,
        #         p3 VARCHAR(255) NOT NULL,
        #         p4 VARCHAR(255) NOT NULL,
        #         p0 VARCHAR(255) NOT NULL,
        #         overflow BOOLEAN NOT NULL,
        #         other_object BOOLEAN NOT NULL,
        #         image_time TIMESTAMP NOT NULL,
        #         detect_result BOOLEAN NOT NULL,
        #         object_type VARCHAR(255) NOT NULL,
        #         confidence INT NOT NULL

        self.cursor.execute(commands)
        print('Table_agv created sucessfully.')


    def insert_database(self, data):
        for j in data:
            self.cursor.execute("INSERT INTO agv (camera_id, img_filename, p1, p2, p3, p4, p0, other_object, image_time, detect_result, object_type, confidence) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"
            , (j['camera_id'], j['img_filename'], j['p1'], j['p2'], j['p3'], j['p4'], j['p0'], j['other_object'], j['image_time'], j['detect_result'], j['object_type'], j['confidence']))        
        #print('data insert sucess')


    def open_file(save_dir):
        data = []
        path = f"C:/Users/user/Desktop/yolo_box/yolov5/{save_dir}/location_center/"
        dirlist = os.listdir( path )
        for i in dirlist:
            with open(f"{path}{i}", 'r') as  f:
                file = json.load(f)
                data.append(file)

        return data


    def sleep_time(hour, min, sec):
        print('process excuting...' + str(datetime.now()))
        return hour * 3600 + min *60 +sec
