##!/usr/bin/python
import os, sys, tarfile

def extract(tar_url, extract_path='.'):
    os.chdir(r"E:\emails")
    print (tar_url)
    tar = tarfile.open(tar_url, 'r')
    for item in tar:
        tar.extract(item, extract_path)
        if item.name.find(".tgz") != -1 or item.name.find(".tar") != -1:
            extract(item.name, "./" + item.name[:item.name.rfind('/')])
try:

    extract(r"enron_mail_20150507.tgz")
    print ('Done.')
except:
    # name = os.path.basename(sys.argv[0])
    print  ("err")
    # print name[:name.rfind('.')], '<filename>'