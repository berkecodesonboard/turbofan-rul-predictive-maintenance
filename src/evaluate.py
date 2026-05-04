import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------
# REGRESSION METRİKLERİ
# ---------------------------------------------------------
# Bu projede model bir sınıf tahmin etmiyor.
# Model sayısal bir değer tahmin ediyor:
#
# RUL = Remaining Useful Life
# RUL = kalan faydalı ömür
#
# Örnek:
# Gerçek RUL  = 50 cycle
# Tahmin RUL  = 42 cycle
# Hata        = 8 cycle
#
# Bu yüzden accuracy kullanma.
# Accuracy daha çok classification problemleri içindir.
#
# Burada regression metrikleri kullan:
#
# MAE
# RMSE
# R2
# ---------------------------------------------------------


def regression_metrics(y_true, y_pred):
    """
    Regression modeli için temel hata metriklerini hesapla.

    Parametreler:
        y_true:
            Gerçek RUL değerleri.
            Örnek: [50, 80, 20]

        y_pred:
            Modelin tahmin ettiği RUL değerleri.
            Örnek: [42, 76, 30]

    Dönen değer:
        MAE, RMSE ve R2 değerlerini içeren sözlük.
    """

    # -----------------------------------------------------
    # 1) MAE - Mean Absolute Error
    # -----------------------------------------------------
    # Ortalama mutlak hata olarak düşün.
    #
    # Her tahmin için hatayı hesapla:
    #
    # hata = gerçek_değer - tahmin_değeri
    #
    # Sonra mutlak değer al.
    # Çünkü -8 hata ile +8 hata aynı büyüklükte hatadır.
    #
    # Örnek:
    # Gerçek RUL = 50
    # Tahmin RUL = 42
    #
    # Hata = 50 - 42 = 8
    #
    # MAE sana şunu söyler:
    # "Model ortalama kaç cycle yanılıyor?"
    #
    # Bu projede en kolay yorumlanan metrik MAE'dir.
    mae = mean_absolute_error(y_true, y_pred)

    # -----------------------------------------------------
    # 2) RMSE - Root Mean Squared Error
    # -----------------------------------------------------
    # Kök ortalama kare hata olarak düşün.
    #
    # RMSE büyük hataları MAE'ye göre daha sert cezalandırır.
    #
    # Neden?
    # Çünkü önce hataların karesini alır.
    #
    # Örnek:
    # 5 cycle hata  -> 25
    # 50 cycle hata -> 2500
    #
    # Büyük hata kare alma işleminden dolayı çok büyür.
    #
    # Bu projede RMSE önemli.
    # Çünkü uçak motoru gibi kritik sistemlerde çok büyük hata isteme.
    #
    # Model genelde iyi tahmin yapıp birkaç motorda aşırı kötü tahmin yapıyorsa,
    # RMSE bunu daha net gösterir.
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # -----------------------------------------------------
    # 3) R2 - R-Squared Score
    # -----------------------------------------------------
    # Modelin hedef değişkeni ne kadar açıklayabildiğini gösterir.
    #
    # Basit yorum:
    #
    # R2 = 1.0  -> mükemmel tahmin
    # R2 = 0.0  -> model ortalama değer söylemekten daha iyi değil
    # R2 < 0.0  -> model çok kötü, ortalama tahmin bile daha iyi olabilir
    #
    # R2 tek başına yeterli değildir.
    # Bu projede R2'yi MAE ve RMSE ile birlikte yorumla.
    r2 = r2_score(y_true, y_pred)

    # -----------------------------------------------------
    # SONUCU SÖZLÜK OLARAK DÖNDÜR
    # -----------------------------------------------------
    # float(...) kullan.
    # Çünkü bazı metrikler numpy.float64 olarak dönebilir.
    # CSV'ye yazarken ve terminalde okurken normal Python float daha temiz görünür.
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }