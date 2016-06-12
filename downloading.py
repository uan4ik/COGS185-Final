import os
import urllib.request
import urllib
import shutil
import imghdr
import os.path



path = "10_classes/"
# for AWS path = "10_classes/"
files = os.listdir(path)




def download():
    for file in files:
        print(file)
        if os.path.isdir("all_10_classes/" + file[:-4]) is False:
            os.makedirs("all_10_classes/" + file[:-4])
        else:
            print(file, " already exists, skipping")
            continue
        count = 0
        line = 0
        f = open(path + file)
        for url in f:
            line += 1
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                file_name = "all_10_classes/" + file[:-4] + "/" + file[:-4] + "_" + str(count)
                response = urllib.request.urlopen(req, timeout=2)
                out_file = open(file_name, 'wb')
                shutil.copyfileobj(response, out_file)
                count += 1
            except:
                print("bad format, or impossible to download line -> ", line)
                continue
            if os.path.isfile(file_name):
                if imghdr.what(file_name) != "jpeg":
                    print(imghdr.what(file_name))
                    if (imghdr.what(file_name)) is None:
                        os.remove(file_name)


            print(line, " saving image # ", count)



'''






'''



download()



