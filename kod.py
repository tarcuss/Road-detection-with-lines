import cv2
import numpy as np

def serit_filtresi(giris_resim, tau):

    # Çıkış resmini oluştur
    cikis_resim = np.zeros_like(giris_resim, dtype=np.uint8)

    # Şerit filtresini uygula
    satir, sutun = v_kanali.shape
    for i in range(tau, satir - tau):
        for j in range(tau, sutun - tau):
            # Denklemi uygula
            deger = 2 * v_kanali[i, j] - (v_kanali[i - tau, j] + v_kanali[i + tau, j]) - abs(v_kanali[i - tau, j] - v_kanali[i + tau, j])

            # Çıkış resmini güncelle
            cikis_resim[i, j] = np.clip(deger, 0, 255)

    return cikis_resim

def extend_line(rho, theta, length_factor):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 - length_factor * (-b))
    y1 = int(y0 - length_factor * (a))
    x2 = int(x0 + length_factor * (-b))
    y2 = int(y0 + length_factor * (a))
    return x1, y1, x2, y2



# Giriş resmini yükle
giris_resim = cv2.imread('image4.jpg')

# Giriş resmini HSV uzayına dönüştür
hsv_resim = cv2.cvtColor(giris_resim, cv2.COLOR_BGR2HSV)

# V (Value) kanalını al
v_kanali = hsv_resim[:, :, 2]
cv2.imshow("hsv uzayi",v_kanali)

##### Şerit filtresini uygula #####
tau = 0  # Şerit kalınlığı
cikis_resim = serit_filtresi(v_kanali, tau)

cv2.imshow("serit filteresi",cikis_resim)

# Otsu eşikleme yöntemi ile eşik değerini bul
_, otsu_image = cv2.threshold(cikis_resim, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imshow("otsu",otsu_image)
######## Hough dönüşümü ile çizgileri belirleme #######

# Hough dönüşümü uygula
kenarlar = cv2.Canny(otsu_image, 50, 150)


# Belirli açı (55.5 ve 40.5) için Hough dönüşümü uygula

lines = cv2.HoughLines(kenarlar, 1, np.deg2rad(55.5) , threshold=100)

# Hough dönüşümü sonuçlarını çiz
sonuc_resim = np.copy(giris_resim)
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho

    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    # Çizgileri çiz
    cv2.line(sonuc_resim, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Kırmızı renk

# İkinci Hough dönüşümü
lines2 = cv2.HoughLines(kenarlar, 1, np.deg2rad(40.5) , threshold=100)

# İkinci Hough dönüşümü sonuçlarını çiz
for line in lines2:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    # Çizgileri çiz
    cv2.line(sonuc_resim, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Yeşil renk

cv2.imshow('seritcizgi', sonuc_resim)

##### UFUK NOKTASI BULMA ######
# İki çizginin kesişim noktasını bul
rho1, theta1 = lines[0][0]
rho2, theta2 = lines2[0][0]
A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
b = np.array([rho1, rho2])
kesisim_noktasi = np.linalg.solve(A, b)

# Kesişim noktasını işaretle
cv2.circle(sonuc_resim, (int(kesisim_noktasi[0]), int(kesisim_noktasi[1])), 10, (255, 255, 255), -1)




######### Cizgileri uzat ######
extended_lines = []
for line in lines:
    rho, theta = line[0]
    x1, y1, x2, y2 = extend_line(rho, theta, length_factor=1400)
    cv2.line(sonuc_resim, (x1, y1), (x2, y2), (0, 0, 255), 2)
    extended_lines.append((rho, theta))

for line in lines2:
    rho, theta = line[0]
    x1, y1, x2, y2 = extend_line(rho, theta, length_factor=1400)
    cv2.line(sonuc_resim, (x1, y1), (x2, y2), (0, 255, 0), 2)
    extended_lines.append((rho, theta))

cv2.imshow('çizgi ve noktalari belirle', sonuc_resim)



# Histogramları oluştur
point_of_intersection = np.array([0.0, 0.0])
for line1, line2 in zip(extended_lines, extended_lines[1:]):
    rho1, theta1 = line1
    rho2, theta2 = line2
    A = np.array([[np.cos(theta1), np.sin(theta1)],
                  [np.cos(theta2), np.sin(theta2)]])
    b = np.array([rho1, rho2])
    intersection = np.linalg.solve(A, b)
    point_of_intersection += intersection

hist1 = cv2.calcHist([cikis_resim], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([sonuc_resim], [0], None, [256], [0, 256])

# Histogramları normalize et
cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

# Bhattacharyya uzaklığını hesapla
bhattacharyya_distance = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

# Konsola Bhattacharyya uzaklığını yazdır
print(f"Bhattacharyya Uzaklığı: {bhattacharyya_distance}")

# Bhattacharyya uzaklığına göre çizgiyi doldur
if bhattacharyya_distance > 0.7:
    # İlk çizginin uzantısının altındaki bölgeyi doldur

    for line in lines2:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x1, y1, x2, y2 = extend_line(rho, theta, length_factor=1400)
        cv2.fillPoly(sonuc_resim, [
            np.array([[x1 - 60, 720], [1410 + x2, 720], [int(kesisim_noktasi[0]), int(kesisim_noktasi[1])]])],
                     (0, 255, 0))
# Sonucu göster
cv2.imshow('yolu doldur', sonuc_resim)


cv2.waitKey(0)
cv2.destroyAllWindows()