import os
import cv2

root = "asl_train/"
folders = os.listdir(root)
folders.sort()

print(folders)
for folder in folders:
    files = os.listdir(root+folder)
    print(folder, len(files))
    x = 0
    for file in files:
        x+=1
        
        if x==50:
            break
        # print(root + file)
        
        img = cv2.imread(root+folder+"/" + file)
        

        img = cv2.resize(img, (128,128), interpolation = cv2.INTER_AREA)
        
        # Save Image
        cv2.imwrite("./data/val/"+folder+"/"+str(x)+".jpg", img)
