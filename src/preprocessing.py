import pandas as pd
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# MODELİN GÖRECEĞİ GİRDİ KOLONLARI
# ---------------------------------------------------------
# Modelin tahmin yaparken hangi kolonlara bakacağını burada belirle.
#
# unit_id kolonunu girdi olarak kullanma.
# Çünkü unit_id sadece motor numarasıdır.
# Motor numarası, fiziksel bir sensör bilgisi değildir.
#
# cycle kolonunu da şimdilik girdi olarak kullanma.
# Çünkü ilk baseline modelde sensörler ve operasyonel ayarlar üzerinden tahmin yap.
#
# Girdi olarak şunları kullan:
# - 3 operational setting
# - 21 sensor
# ---------------------------------------------------------


FEATURE_COLUMNS = (
    ["setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)


# ---------------------------------------------------------
# TRAIN VERİSİNE RUL EKLE
# ---------------------------------------------------------
# Train verisinde motorlar bozulana kadar çalıştırılmıştır.
# Bu yüzden her motorun son cycle değeri, o motorun failure noktası kabul edilir.
#
# RUL formülü:
#
# RUL = motorun_son_cycle_değeri - mevcut_cycle
#
# Örnek:
# Bir motor 200. cycle'da bozuluyorsa:
#
# cycle 1   -> RUL = 199
# cycle 100 -> RUL = 100
# cycle 200 -> RUL = 0
# ---------------------------------------------------------


def add_train_rul(train_df, cap=None):
    """
    Train tablosuna RUL kolonunu ekle.

    Parametreler:
        train_df:
            Ham train tablosu.

        cap:
            RUL için üst sınır.
            Örnek: cap=125 verirsen 125'ten büyük RUL değerlerini 125 yap.

    Dönen değer:
        RUL kolonu eklenmiş yeni DataFrame.
    """

    # Orijinal tabloyu doğrudan değiştirme.
    # Önce kopyasını al.
    df = train_df.copy()

    # Her motorun ulaştığı en büyük cycle değerini bul.
    #
    # groupby("unit_id"):
    # Tabloyu motor numarasına göre grupla.
    #
    # transform("max"):
    # Her motor için maksimum cycle değerini bul
    # ve o motorun tüm satırlarına yaz.
    df["max_cycle"] = df.groupby("unit_id")["cycle"].transform("max")

    # Her satır için kalan ömrü hesapla.
    #
    # max_cycle:
    # Motorun bozulduğu kabul edilen cycle.
    #
    # cycle:
    # Şu anki cycle.
    #
    # RUL:
    # Bozulmaya kaç cycle kaldığı.
    df["RUL"] = df["max_cycle"] - df["cycle"]

    # Capping uygula.
    #
    # Motor çok sağlıklıyken 250 cycle mı kaldı, 220 cycle mı kaldı ayrımı
    # model için çok kritik değildir.
    #
    # Asıl kritik bölge, motor failure'a yaklaşırken başlar.
    # Bu yüzden büyük RUL değerlerini belirli bir üst sınıra sabitle.
    #
    # Örnek:
    # cap = 125 ise:
    # 180 -> 125 olur
    # 130 -> 125 olur
    # 90  -> 90 kalır
    if cap is not None:
        df["RUL"] = df["RUL"].clip(upper=cap)

    return df


# ---------------------------------------------------------
# TEST VERİSİNE RUL EKLE
# ---------------------------------------------------------
# Test verisi train verisinden farklıdır.
#
# Train:
# Motor failure'a kadar gider.
#
# Test:
# Motor failure olmadan önce kesilir.
#
# Bu yüzden test verisinde RUL üretmek için RUL_FDxxx.txt dosyasını kullan.
#
# Test RUL mantığı:
#
# failure_cycle = testteki_son_cycle + final_RUL
# RUL = failure_cycle - mevcut_cycle
#
# Örnek:
# Test motoru 31. cycle'da kesilmiş olsun.
# RUL dosyası bu motor için final_RUL = 112 desin.
#
# failure_cycle = 31 + 112 = 143
#
# cycle 1  -> RUL = 142
# cycle 31 -> RUL = 112
# ---------------------------------------------------------


def add_test_rul(test_df, test_rul_df, cap=None):
    """
    Test tablosuna RUL kolonunu ekle.

    Parametreler:
        test_df:
            Ham test tablosu.

        test_rul_df:
            RUL_FDxxx.txt dosyasından gelen cevap tablosu.
            İçinde unit_id ve final_RUL kolonları olmalı.

        cap:
            RUL için üst sınır.
            Örnek: cap=125.

    Dönen değer:
        RUL kolonu eklenmiş yeni test DataFrame.
    """

    # Orijinal test tablosunu doğrudan değiştirme.
    # Önce kopyasını al.
    df = test_df.copy()

    # Her test motorunun gözlenen son cycle değerini bul.
    #
    # Test verisi failure'a kadar gitmediği için bu değer failure cycle değildir.
    # Sadece elimizde görünen son cycle değeridir.
    max_test_cycles = (
        df.groupby("unit_id")["cycle"]
        .max()
        .reset_index()
        .rename(columns={"cycle": "max_test_cycle"})
    )

    # max_test_cycle bilgisini ana test tablosuna ekle.
    #
    # merge:
    # unit_id üzerinden iki tabloyu birleştir.
    df = df.merge(max_test_cycles, on="unit_id", how="left")

    # RUL cevap dosyasından gelen final_RUL bilgisini de ekle.
    #
    # final_RUL:
    # Test motorunun son gözlenen cycle'dan sonra kaç cycle daha çalışacağını gösterir.
    df = df.merge(test_rul_df, on="unit_id", how="left")

    # Test motorunun gerçek failure cycle değerini hesapla.
    #
    # failure_cycle:
    # Motorun toplamda bozulacağı cycle.
    df["failure_cycle"] = df["max_test_cycle"] + df["final_RUL"]

    # Her satır için RUL hesapla.
    df["RUL"] = df["failure_cycle"] - df["cycle"]

    # Train verisinde yaptığın capping işlemini test verisine de uygula.
    # Train ve test hedeflerinin aynı mantıkla hazırlanmasına dikkat et.
    if cap is not None:
        df["RUL"] = df["RUL"].clip(upper=cap)

    return df


# ---------------------------------------------------------
# FEATURE VE TARGET AYIR
# ---------------------------------------------------------
# Makine öğrenmesinde tabloyu iki parçaya ayır:
#
# X:
# Modelin göreceği girdiler.
# Burada setting ve sensor kolonlarını kullan.
#
# y:
# Modelin tahmin etmeye çalıştığı hedef.
# Burada RUL kolonunu kullan.
# ---------------------------------------------------------


def split_features_target(df):
    """
    Tabloyu X ve y olarak ayır.

    X:
        Model girdileri.

    y:
        Model hedefi.
    """

    # Modelin göreceği kolonları seç.
    X = df[FEATURE_COLUMNS]

    # Modelin tahmin etmeye çalışacağı hedefi seç.
    y = df["RUL"]

    return X, y


# ---------------------------------------------------------
# TRAIN VE TEST VERİSİNİ ÖLÇEKLE
# ---------------------------------------------------------
# Sensörlerin değer aralıkları birbirinden çok farklı olabilir.
#
# Örnek:
# sensor_1 değeri 500 civarında olabilir.
# sensor_2 değeri 0.02 civarında olabilir.
#
# Bazı modeller bu farklı ölçeklerden etkilenebilir.
# Bu yüzden veriyi aynı ölçeğe getir.
#
# StandardScaler:
# Her kolonu yaklaşık şu hale getirir:
#
# ortalama = 0
# standart sapma = 1
#
# En önemli kural:
# Scaler'ı sadece train veriye fit et.
# Test verisine fit etme.
#
# Çünkü test verisini eğitim sırasında görmek data leakage olur.
# ---------------------------------------------------------


def scale_train_test(X_train, X_test):
    """
    Train ve test feature tablolarını ölçekle.

    Dikkat:
        Scaler'ı sadece train veriye fit et.
        Test verisini sadece transform et.
    """

    # StandardScaler nesnesi oluştur.
    scaler = StandardScaler()

    # Train veriden ölçekleme parametrelerini öğren.
    # Sonra train veriyi dönüştür.
    #
    # fit:
    # Train verinin ortalama ve standart sapmasını öğren.
    #
    # transform:
    # Öğrenilen değerlere göre veriyi dönüştür.
    X_train_scaled = scaler.fit_transform(X_train)

    # Test verisine fit uygulama.
    # Sadece train'den öğrenilen scaler ile dönüştür.
    #
    # Bu adım data leakage'i engeller.
    X_test_scaled = scaler.transform(X_test)

    # Scaler'ı da döndür.
    # Çünkü last-cycle test gibi başka test parçalarını aynı scaler ile dönüştürmek gerekir.
    return X_train_scaled, X_test_scaled, scaler