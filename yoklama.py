import tkinter as tk
from tkinter import Message ,Text
import cv2,os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font

window1 = tk.Tk()
window1.title("Gerçek Zamanlı Yoklama")
window1.geometry('600x300')
window1.configure(background='white')
message1 = tk.Label(window1, text="Gerçek Zamanlı Yoklama"  ,fg="black", bg="white"  ,font=('System', 25)) 
message1.place(x=160, y=35)


def DispWin():
    
    window = tk.Tk()
    window.title("Öğrenci Kayıt")

    dialog_title = 'ÇIKIŞ'
    dialog_text = 'Çıkmak istediğinizden emin misiniz?'
 
    window.geometry('1280x720')
    window.configure(background='white')


    window.grid_rowconfigure(0, weight=1)
    window.grid_columnconfigure(0, weight=1)


    message = tk.Label(window, text="Öğrenci Kayıt Ekranı"  ,fg="black", bg="white"  ,font=('System', 26)) 
    
    message.place(x=200, y=50)

    lbl = tk.Label(window, text="Öğrenci Numarası",width=20  ,height=2, bg="white"  ,font=('System', 15, ' bold ') ) 
    lbl.place(x=200, y=200)

    txt = tk.Entry(window,width=20  ,bg="white" ,font=('arial', 15, ' bold '))
    txt.place(x=600, y=215)

    lbl2 = tk.Label(window, text="Öğrenci Adı Soyadı",width=20  ,bg="white"    ,height=2 ,font=('System', 15, ' bold ')) 
    lbl2.place(x=200, y=300)

    txt2 = tk.Entry(window,width=20  ,bg="white"  ,font=('arial', 15, ' bold ')  )
    txt2.place(x=600, y=315)

    lbl3 = tk.Label(window, text="Durum : ",width=20  ,bg="white"  ,height=2 ,font=('System', 15, ' bold ')) 
    lbl3.place(x=200, y=400)

    message = tk.Label(window, text="" ,bg="white"  ,width=50  ,height=2, activebackground = "blue" ,font=('System', 15, ' bold ')) 
    message.place(x=600, y=400)
    

    

    
    
 
   
    
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            pass
 
        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass
 
        return False
 
    def TakeImages():        
        Id=(txt.get())
        name=(txt2.get())
        if(is_number(Id) and name != ""):
            cam = cv2.VideoCapture(0)
            harcascadePath = "haarcascade_frontalface_default.xml"
            detector=cv2.CascadeClassifier(harcascadePath)
            sampleNum=0
            while(True):
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)
                for (x,y,w,h) in faces:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        
                    sampleNum=sampleNum+1
                    cv2.imwrite("TrainingImage/ "+name +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                    cv2.imshow('Kamera',img)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
                elif sampleNum>60:
                    break
            cam.release()
            cv2.destroyAllWindows() 
            res = "" + name +" için görseller kaydedildi."
            row = [Id , name]
            with open('OgrenciBilgileri.csv','a+') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)
            csvFile.close()
            message.configure(text= res)
        else:
            if(is_number(Id)):
                res = "Yalnızca harf girişi yapabilirsiniz."
                message.configure(text= res)
            if(name.isalpha()):
                res = "Yalnızca sayı girişi yapabilirsiniz."
                message.configure(text= res)

    def TrainImages():
        recognizer = cv2.face_LBPHFaceRecognizer.create()
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector =cv2.CascadeClassifier(harcascadePath)
        faces,Id = getImagesAndLabels("TrainingImage")
        recognizer.train(faces, np.array(Id))
        recognizer.save("trainner.yml")
        res = "Eğitim Tamamlandı."
        message.configure(text= res)

    
    


    def getImagesAndLabels(path):
        imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
        faces=[]
        Ids=[]
        for imagePath in imagePaths:
            pilImage=Image.open(imagePath).convert('L')
            imageNp=np.array(pilImage,'uint8')
            Id=int(os.path.split(imagePath)[-1].split(".")[1])
            faces.append(imageNp)
            Ids.append(Id)        
        return faces,Ids

    
        
    takeImg = tk.Button(window, text="Resim Çek", command=TakeImages  ,width=15  ,height=2, activebackground = "Red" ,font=('System', 17, ' bold '))
    takeImg.place(x=200, y=500)
    trainImg = tk.Button(window, text="Resimleri İşle", command=TrainImages  ,width=15  ,height=2, activebackground = "Red" ,font=('System', 17, ' bold '))
    trainImg.place(x=500, y=500)
    quitWindow = tk.Button(window, text="Kapat", command=window.destroy  ,width=12  ,height=2, activebackground = "Red" ,font=('System', 17, ' bold '))
    quitWindow.place(x=800, y=500)
    window.mainloop()

def TrackImages():
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("trainner.yml")
        harcascadePath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(harcascadePath);    
        df=pd.read_csv("OgrenciBilgileri.csv")
        cam = cv2.VideoCapture(0)
        font = cv2.FONT_HERSHEY_SIMPLEX        
        col_names =  ['Id','name','date','time']
        attendance = pd.DataFrame(columns = col_names)    
        while True:
            ret, im =cam.read()
            gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            faces=faceCascade.detectMultiScale(gray, 1.2,5)    
            for(x,y,w,h) in faces:
                cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
                Id, conf = recognizer.predict(gray[y:y+h,x:x+w])                                   
                if(conf < 60):
                    ts = time.time()      
                    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                    aa=df.loc[df['Id'] == Id]['Name'].values
                    tt=str(Id)+"-"+aa
                    attendance.loc[len(attendance)] = [Id,aa,date,timeStamp]
                
                else:
                    Id='BILINMEYEN OGRENCI'                
                    tt=str(Id)  
                        
                cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)        
            attendance=attendance.drop_duplicates(subset=['Id'],keep='first')    
            cv2.imshow('YOKLAMAYI BITIRMEK ICIN Q HARFINE BASINIZ',im) 
            if (cv2.waitKey(1)==ord('q')):
                break
        ts = time.time()      
        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        Hour,Minute,Second=timeStamp.split(":")
        fileName="Yoklama/Yoklama_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
        attendance.to_csv(fileName,index=False)
        cam.release()
        cv2.destroyAllWindows()


Attend = tk.Button(window1, text="Yoklama Al", command=TrackImages  , width=10  ,height=2 ,activebackground = "Red" ,font=('System', 15, ' bold '))
Attend.place(x=110, y=140)
Register = tk.Button(window1, text="Öğrenci Kayıt Et", command=DispWin  , width=20  ,height=2 ,activebackground = "Red" ,font=('System', 15, ' bold '))
Register.place(x=260, y=140)



window1.mainloop()
 

