'''
Author: xuhy 1727317079@qq.com
Date: 2023-07-23 22:06:30
LastEditors: xuhy 1727317079@qq.com
LastEditTime: 2023-08-16 09:59:52
FilePath: \deeplabv3_plus-voyager\png\jpeg_to_jpg.py
Description: png-->jpg
'''
import os
dirName = "../datasetsCoarse/JPEGImages\\"        
li=os.listdir(dirName)

if __name__ == "__main__":
    for filename in li:
        newname = filename
        newname = newname.split(".")
        if newname[-1]=="jpeg":
            newname[-1]="jpg"
            newname = str.join(".", newname)  
            filename = dirName+filename
            newname = dirName+newname
            os.rename(filename,newname)
            print(newname,"updated successfully")
        if newname[-1]=="png":
            newname[-1]="jpg"
            newname = str.join(".", newname) 
            filename = dirName+filename
            newname = dirName+newname
            os.rename(filename,newname)
            print(newname,"updated successfully")
