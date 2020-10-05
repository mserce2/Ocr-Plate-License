from pymetehan.anpr import PyMetehanSerceANPR
from imutils import paths
import argparse
import imutils
import cv2

def cleanup_text(text):
    #ASCII olmayan metni çıkarın, böylece OpenCV kullanarak resimdeki metni daha doğru kullanırız
    #yani örneğin Ğ harfi gibi değişik karakteler bulursak resim üzerinde opencv otomatik
    #olarak "?" işareti atayacak.Ve görüntü metnimiz kötü görümncek bunu engellemek için
    #bu fonksiyonu kullancağız
    return "".join([c if ord(c) <128 else "" for c in text]).strip()

#Bağımsız değişkenlerimizi yapılandırıyoruz

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input directory of images") #Girdi görüntülerinin dosya yolu
ap.add_argument("-c", "--clear-border", type=int, default=-1,
	help="whether or to clear border pixels before OCR'ing") #OCR'lamadan önce kenar piksellerini temizleyip temizlemeyeceği
ap.add_argument("-p", "--psm",type=int,default=7,
                help="psm mode for OCR'ing license plates") #OCR'lama plakaları için varsayılan PSM modu
ap.add_argument("-d", "--debug",type=int,default=-1,
                help="default PSM mode for OCR'ing license plates") #ek görselleştirmelerin gösterilip gösterilmeyeceği
args = vars(ap.parse_args())

#İçe aktarmalarımız, tanımlanmış metin temizleme aracı ve komut satırı argümanlarımızın anlaşılmasıyla,
#artık plakaları otomatik olarak tanımanın zamanı geldi!

#ANPR sınıfımızı başlatın
anpr=PyMetehanSerceANPR(debug=args["debug"] > 0)

#giriş dizinindeki tüm görüntü yollarını yakala
imagePaths=sorted(list(paths.list_images(args["input"])))
#Her bir plakayı başarılı bir şekilde bulmak ve OCR'lamak umuduyla, imagePath'lerimizin her birini işleyeceğiz:

#giriş dizinindeki tüm görüntü yolları üzerinde döngü yapıyoruz:
for imagePath in imagePaths:
    #giriş görüntüsünü diskten yükleyin ve yeniden boyutlandırıyoruz
    image=cv2.imread(imagePath)
    image=imutils.resize(image,width=600)

    #otomatik plaka tanımayı uyguluyoruz
    (lpText,lpCnt)=anpr.find_and_ocr(image,psm=args["psm"],
                                     clearBorder=args["clear_border"] > 0)
    #plaka başarıyla OCR'landıysa devam ediyoruz
    if lpText is not None and lpCnt is not None:
        #Döndürülmüş bir sınırlayıcı kutuyu plaka kenarına yerleştirin ve sınırlayıcı kutuyu plaka üzerine çizin
        box=cv2.boxPoints(cv2.minAreaRect(lpCnt))
        box=box.astype("int")
        cv2.drawContours(image,[box],-1,(0,255,0),2)
        #plaka için normal bir sınırlayıcı kutu hesaplayın ve ardından OCR'lanmış plaka metnini görüntünün üzerine çizin
        (x,y,w,h)=cv2.boundingRect(lpCnt)
        cv2.putText(image,cleanup_text(lpText),(x,y-15),
                    cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,255,0),2)
        #ANPR görüntüsünü göster
        print("[INFO] {}".format(lpText))
        cv2.imshow("Output ANPR",image)
        cv2.waitKey(0)