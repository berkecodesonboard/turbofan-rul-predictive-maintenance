from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# KOLON İSİMLERİ
# ---------------------------------------------------------
# NASA C-MAPSS dosyalarını ham .txt olarak oku.
# Bu dosyalarda kolon başlığı bekleme.
# Sadece sayısal değerler olduğunu varsay.
#
# Bu yüzden kolon isimlerini burada açıkça tanımla.
# ---------------------------------------------------------


# Motor kimliği ve cycle bilgisi.
# unit_id: Hangi motor olduğunu göster.
# cycle: O motorun kaçıncı çalışma adımı olduğunu göster.
INDEX_COLUMNS = ["unit_id", "cycle"]


# Motor çalışma koşulları.
# Bunları sensör gibi düşünme.
# Bunlar motorun hangi operasyonel şartlarda çalıştığını gösterir.
SETTING_COLUMNS = [
    "setting_1",
    "setting_2",
    "setting_3",
]


# 21 adet sensör kolonunu otomatik üret.
# Tek tek sensor_1, sensor_2, ..., sensor_21 yazmak yerine liste oluştur.
#
# range(1, 22) ifadesini 1-21 arası sayılar üretmek için kullan.
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]


# Tüm kolonları doğru sırayla birleştir.
# Dosyayı okurken pandas'a bu sırayı ver.
ALL_COLUMNS = INDEX_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS


# ---------------------------------------------------------
# TRAIN VERİSİNİ OKU
# ---------------------------------------------------------
# Train dosyası motorların bozulana kadar çalıştığı veridir.
# Bu dosyada doğrudan RUL kolonu bekleme.
# RUL değerini daha sonra preprocessing aşamasında üret.
# ---------------------------------------------------------


def load_train_data(data_dir, dataset_id="FD001"):
    """
    Train dosyasını oku.

    Parametreler:
        data_dir:
            Ham veri klasörünü ver.
            Örnek: "data/raw"

        dataset_id:
            Okunacak alt veri setini seç.
            Örnek: "FD001", "FD002", "FD003", "FD004"

    Dönen değer:
        Kolon isimleri verilmiş pandas DataFrame.
    """

    # Klasör yolunu Path nesnesine çevir.
    # Böylece Windows/Linux yol farklarını daha rahat yönet.
    data_dir = Path(data_dir)

    # Okunacak train dosyasının yolunu oluştur.
    # Örnek:
    # dataset_id = "FD001" ise dosya adı train_FD001.txt olur.
    file_path = data_dir / f"train_{dataset_id}.txt"

    # txt dosyasını pandas ile oku.
    #
    # sep=r"\s+":
    # Bir veya daha fazla boşluğu ayırıcı kabul et.
    #
    # header=None:
    # Dosyada kolon başlığı olmadığını söyle.
    #
    # names=ALL_COLUMNS:
    # Kolon isimlerini dışarıdan ver.
    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=ALL_COLUMNS,
    )

    return df


# ---------------------------------------------------------
# TEST VERİSİNİ OKU
# ---------------------------------------------------------
# Test dosyası motorlar bozulmadan önce kesilmiş veridir.
# Bu dosyada da doğrudan RUL kolonu bekleme.
# RUL bilgisini ayrı RUL_FDxxx.txt dosyasından üret.
# ---------------------------------------------------------


def load_test_data(data_dir, dataset_id="FD001"):
    """
    Test dosyasını oku.

    Test dosyası sadece gözlenen cycle'ları içerir.
    Motorun failure noktasını doğrudan içermez.
    """

    # Klasör yolunu Path nesnesine çevir.
    data_dir = Path(data_dir)

    # Okunacak test dosyasının yolunu oluştur.
    # Örnek:
    # dataset_id = "FD001" ise dosya adı test_FD001.txt olur.
    file_path = data_dir / f"test_{dataset_id}.txt"

    # Test verisini train verisiyle aynı kolon isimleriyle oku.
    # Böylece preprocessing aşamasında iki tabloyu aynı formatta kullan.
    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=ALL_COLUMNS,
    )

    return df


# ---------------------------------------------------------
# TEST RUL CEVAP DOSYASINI OKU
# ---------------------------------------------------------
# RUL_FDxxx.txt dosyasını test verisinin cevap anahtarı gibi düşün.
# Bu dosya her test motorunun son gözlenen cycle'ında kaç cycle ömrü kaldığını verir.
#
# Dosyada unit_id kolonu yoktur.
# Satır sırasını motor numarası olarak kullan.
# ---------------------------------------------------------


def load_test_rul(data_dir, dataset_id="FD001"):
    """
    Test motorlarının final RUL değerlerini oku.

    Örnek:
        RUL_FD001.txt içindeki 1. satır -> unit_id = 1
        RUL_FD001.txt içindeki 2. satır -> unit_id = 2
    """

    # Klasör yolunu Path nesnesine çevir.
    data_dir = Path(data_dir)

    # Okunacak RUL dosyasının yolunu oluştur.
    # Örnek:
    # dataset_id = "FD001" ise dosya adı RUL_FD001.txt olur.
    file_path = data_dir / f"RUL_{dataset_id}.txt"

    # RUL dosyasında tek kolon vardır.
    # Bu kolona final_RUL adını ver.
    rul_df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=["final_RUL"],
    )

    # RUL dosyasında motor numarası yoktur.
    # Satır sırasından unit_id üret.
    #
    # len(rul_df) kadar motor olduğunu kabul et.
    # unit_id değerlerini 1'den başlat.
    rul_df["unit_id"] = range(1, len(rul_df) + 1)

    # Kolon sırasını temiz tut.
    return rul_df[["unit_id", "final_RUL"]]