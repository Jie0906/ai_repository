from distutils.log import error
from msilib.schema import Error
from turtle import title
import psycopg2
import json
import os
from threading import Timer
from datetime import datetime
import time
from detect import main, parse_opt
from db_related import DbHandler
from dotenv import load_dotenv



if __name__ == '__main__':
    try:
        #db_config
        load_dotenv()
        host = os.getenv('DB_HOST')
        port = os.getenv('DB_PORT')
        dbname = os.getenv('DB_NAME')
        user = os.getenv('DB_USER')
        password = os.getenv('DB_PASSWORD')
        conn_string = 'host = {} port = {} user = {} dbname = {} password = {}'.format(host, port, user, dbname, password)
        conn = psycopg2.connect(conn_string)
        conn.autocommit = True
        print("DB connected sucessfully!")

        cursor = conn.cursor()
        second = DbHandler.sleep_time(0,0,5)
        

        #程式邏輯:
        # 1.建立table(若已存在則忽略)
        # 2.進行預測並寫入資料庫
        # 3.睡眠5秒
        
        while True:
            DbHandler(cursor).create_table()
            opt = parse_opt()
            main(cursor, opt)
            time.sleep(second)

            
        
    except Exception as e:
        print("Error: ",e)
        cursor.close()




