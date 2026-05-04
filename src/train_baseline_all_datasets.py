from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from data_loader import load_test_data, load_test_rul, load_train_data
from evaluate import regression_metrics
from plots import (
    plot_actual_vs_predicted,
    plot_error_histogram,
    plot_model_comparison,
)
from preprocessing import (
    add_test_rul,
    add_train_rul,
    scale_train_test,
    split_features_target,
)


# ---------------------------------------------------------
# PROJE KLASÖR YOLLARI
# ---------------------------------------------------------
# Bu dosya src/ klasörünün içinde duruyor.
#
# Path(__file__)                  -> bu dosyanın yolunu verir
# Path(__file__).resolve()        -> tam dosya yolunu verir
# Path(__file__).resolve().parents[1] -> proje ana klasörüne çıkar
#
# Örnek:
# src/train_baseline_all_datasets.py
# buradan bir üst seviye proje ana klasörüdür.
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "raw"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"


# ---------------------------------------------------------
# RUL ÜST SINIRI
# ---------------------------------------------------------
# Çok büyük RUL değerlerini sınırlamak için kullan.
#
# Örnek:
# Gerçek RUL = 250 ise bunu 125 yap.
# Gerçek RUL = 180 ise bunu 125 yap.
# Gerçek RUL = 90 ise 90 olarak bırak.
#
# Neden?
# Motor çok sağlıklıyken 250 cycle mı kaldı, 220 cycle mı kaldı ayrımı çok kritik değildir.
# Asıl kritik bölge motor failure'a yaklaşırken başlar.
# ---------------------------------------------------------

RUL_CAP = 125


# ---------------------------------------------------------
# ÇALIŞTIRILACAK DATASET LİSTESİ
# ---------------------------------------------------------
# NASA C-MAPSS içinde dört ana alt veri seti var.
#
# FD001:
# Daha basit senaryo.
#
# FD002:
# Daha zor senaryo.
#
# FD003:
# Farklı fault/degradation yapısı içerir.
#
# FD004:
# Daha karmaşık senaryo.
#
# data/raw klasöründe bu dört dataset için train, test ve RUL dosyaları olmalı.
# ---------------------------------------------------------

DATASETS = ["FD001", "FD002", "FD003", "FD004"]


# ---------------------------------------------------------
# BASELINE MODELLERİ OLUŞTUR
# ---------------------------------------------------------
# Baseline model, ilk karşılaştırma modeli demektir.
#
# Direkt GRU/LSTM gibi deep learning modellerine atlama.
# Önce klasik makine öğrenmesi modellerini çalıştır.
# Sonra deep learning gerçekten daha iyi mi karşılaştır.
# ---------------------------------------------------------


def build_models():
    """
    Kullanılacak baseline regression modellerini oluştur.
    """

    models = {
        # Linear Regression:
        # En basit regression modellerinden biridir.
        # Sensörler ile RUL arasında düz/lineer ilişki arar.
        #
        # Zayıf tarafı:
        # Motor bozulması genelde lineer değildir.
        "Linear Regression": LinearRegression(),

        # Random Forest:
        # Birden fazla decision tree kullanır.
        # Lineer olmayan ilişkileri Linear Regression'a göre daha iyi yakalar.
        #
        # n_estimators:
        # Kaç tane ağaç kurulacağını belirle.
        #
        # max_depth:
        # Ağacın çok fazla derinleşmesini engelle.
        # Çok derin ağaç ezber yapabilir.
        #
        # random_state:
        # Her çalıştırmada benzer sonucu almak için sabit sayı ver.
        #
        # n_jobs=-1:
        # İşlemcinin uygun tüm çekirdeklerini kullan.
        "Random Forest": RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        ),

        # Gradient Boosting:
        # Küçük ağaçları sırayla kurar.
        # Her yeni ağaç önceki hataları azaltmaya çalışır.
        #
        # learning_rate:
        # Her yeni ağacın etkisini kontrol eder.
        #
        # max_depth:
        # Her ağacın karmaşıklığını sınırlar.
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }

    return models


# ---------------------------------------------------------
# TEST VERİSİNDE SADECE SON CYCLE SATIRLARINI AL
# ---------------------------------------------------------
# Test verisinde her motorun birden fazla cycle satırı vardır.
#
# Örnek:
# unit_id = 1 için cycle 1, 2, 3, ..., 31 olabilir.
#
# RUL_FD001.txt dosyasındaki cevap ise bu motorun son gözlenen cycle'ına aittir.
#
# Bu yüzden last-cycle evaluation yaparken her motorun sadece son satırını al.
# ---------------------------------------------------------


def get_last_cycle_rows(test_df):
    """
    Her test motorunun sadece son gözlenen cycle satırını seç.
    """

    # groupby("unit_id"):
    # Veriyi motor motor grupla.
    #
    # ["cycle"].idxmax():
    # Her motor için en büyük cycle değerine sahip satırın index'ini bul.
    #
    # loc[...]:
    # Bu index'lere karşılık gelen satırları al.
    last_cycle_df = test_df.loc[
        test_df.groupby("unit_id")["cycle"].idxmax()
    ].copy()

    return last_cycle_df


# ---------------------------------------------------------
# TEK BİR DATASET İÇİN TÜM BASELINE AKIŞINI ÇALIŞTIR
# ---------------------------------------------------------
# Bu fonksiyon FD001 veya FD002 gibi tek bir dataset'i işler.
#
# Yapılacak işler:
#
# 1. Train/test/RUL dosyalarını oku.
# 2. Train ve test tablolarına RUL ekle.
# 3. X ve y olarak ayır.
# 4. Veriyi ölçekle.
# 5. Modelleri eğit.
# 6. Tahmin yap.
# 7. Metrikleri hesapla.
# 8. Grafikleri kaydet.
# 9. CSV sonuçlarını kaydet.
# ---------------------------------------------------------


def run_one_dataset(dataset_id):
    """
    Tek bir C-MAPSS dataset'i için baseline modelleri çalıştır.

    Örnek:
        dataset_id = "FD001"
    """

    print("\n==============================")
    print(f"Dataset çalıştırılıyor: {dataset_id}")
    print("==============================")

    # Bu dataset için ayrı grafik ve metrik klasörü oluştur.
    #
    # Örnek:
    # outputs/figures/FD001/
    # outputs/metrics/FD001/
    dataset_figures_dir = FIGURES_DIR / dataset_id
    dataset_metrics_dir = METRICS_DIR / dataset_id

    dataset_figures_dir.mkdir(parents=True, exist_ok=True)
    dataset_metrics_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # 1) VERİYİ OKU
    # -----------------------------------------------------
    # data_loader.py içindeki fonksiyonları kullan.
    #
    # train_raw:
    # Eğitim verisi.
    #
    # test_raw:
    # Test verisi.
    #
    # test_rul:
    # Test motorlarının son gözlenen cycle'daki gerçek RUL değerleri.
    # -----------------------------------------------------

    print("Veri okunuyor...")

    train_raw = load_train_data(DATA_DIR, dataset_id)
    test_raw = load_test_data(DATA_DIR, dataset_id)
    test_rul = load_test_rul(DATA_DIR, dataset_id)

    # -----------------------------------------------------
    # 2) RUL ETİKETLERİNİ EKLE
    # -----------------------------------------------------
    # Train verisinde RUL:
    # max_cycle - current_cycle
    #
    # Test verisinde RUL:
    # failure_cycle - current_cycle
    #
    # RUL_CAP kullanarak büyük RUL değerlerini sınırla.
    # -----------------------------------------------------

    print("RUL etiketleri ekleniyor...")

    train_df = add_train_rul(train_raw, cap=RUL_CAP)
    test_df = add_test_rul(test_raw, test_rul, cap=RUL_CAP)

    # -----------------------------------------------------
    # 3) X VE y AYIR
    # -----------------------------------------------------
    # X:
    # Model girdileri.
    # setting_1, setting_2, setting_3, sensor_1 ... sensor_21
    #
    # y:
    # Modelin tahmin edeceği hedef.
    # RUL
    # -----------------------------------------------------

    X_train, y_train = split_features_target(train_df)

    # All-cycle evaluation için test tablosunun tüm satırlarını kullan.
    X_test_all, y_test_all = split_features_target(test_df)

    # Last-cycle evaluation için her motorun sadece son satırını kullan.
    test_last_df = get_last_cycle_rows(test_df)
    X_test_last, y_test_last = split_features_target(test_last_df)

    # -----------------------------------------------------
    # 4) SCALING UYGULA
    # -----------------------------------------------------
    # Scaler'ı sadece train veriye fit et.
    # Test verisine fit etme.
    #
    # Bu kuralı bozma.
    # Aksi halde data leakage oluşur.
    # -----------------------------------------------------

    print("Veri ölçekleniyor...")

    X_train_scaled, X_test_all_scaled, scaler = scale_train_test(
        X_train,
        X_test_all,
    )

    # Last-cycle test verisini de aynı scaler ile dönüştür.
    # Burada yeniden fit yapma.
    X_test_last_scaled = scaler.transform(X_test_last)

    # -----------------------------------------------------
    # 5) MODELLERİ OLUŞTUR
    # -----------------------------------------------------

    models = build_models()

    # Metrikleri listelerde biriktir.
    # Döngü bittikten sonra pandas DataFrame'e çevir.
    all_cycle_results = []
    last_cycle_results = []

    # -----------------------------------------------------
    # 6) HER MODELİ EĞİT VE TEST ET
    # -----------------------------------------------------

    for model_name, model in models.items():
        print(f"\nModel eğitiliyor: {model_name} | Dataset: {dataset_id}")

        # Modeli sadece train verisiyle eğit.
        model.fit(X_train_scaled, y_train)

        # -------------------------------------------------
        # ALL-CYCLE TAHMİNİ
        # -------------------------------------------------
        # Test verisindeki tüm cycle satırları için tahmin yap.
        # Bu değerlendirme modelin genel davranışını görmek için faydalıdır.
        # -------------------------------------------------

        y_pred_all = model.predict(X_test_all_scaled)

        all_metrics = regression_metrics(y_test_all, y_pred_all)
        all_metrics["dataset"] = dataset_id
        all_metrics["model"] = model_name
        all_metrics["evaluation"] = "all_cycles"

        all_cycle_results.append(all_metrics)

        # -------------------------------------------------
        # LAST-CYCLE TAHMİNİ
        # -------------------------------------------------
        # Her test motorunun sadece son gözlenen cycle'ı için tahmin yap.
        # Bu değerlendirme C-MAPSS test mantığına daha yakındır.
        # -------------------------------------------------

        y_pred_last = model.predict(X_test_last_scaled)

        last_metrics = regression_metrics(y_test_last, y_pred_last)
        last_metrics["dataset"] = dataset_id
        last_metrics["model"] = model_name
        last_metrics["evaluation"] = "last_cycle"

        last_cycle_results.append(last_metrics)

        # -------------------------------------------------
        # GRAFİKLERİ KAYDET
        # -------------------------------------------------
        # Dosya adında boşluk kullanma.
        # "Random Forest" -> "random_forest" yap.
        # -------------------------------------------------

        safe_model_name = model_name.lower().replace(" ", "_")

        plot_actual_vs_predicted(
            y_test_all,
            y_pred_all,
            f"{dataset_id} - {model_name} - All Cycles",
            dataset_figures_dir / f"actual_vs_predicted_{safe_model_name}_all_cycles.png",
        )

        plot_error_histogram(
            y_test_all,
            y_pred_all,
            f"{dataset_id} - {model_name} - All Cycles",
            dataset_figures_dir / f"error_histogram_{safe_model_name}_all_cycles.png",
        )

        plot_actual_vs_predicted(
            y_test_last,
            y_pred_last,
            f"{dataset_id} - {model_name} - Last Cycle",
            dataset_figures_dir / f"actual_vs_predicted_{safe_model_name}_last_cycle.png",
        )

        plot_error_histogram(
            y_test_last,
            y_pred_last,
            f"{dataset_id} - {model_name} - Last Cycle",
            dataset_figures_dir / f"error_histogram_{safe_model_name}_last_cycle.png",
        )

        # Terminalde model sonucunu gör.
        print(f"{model_name} all-cycle metrics:")
        print(all_metrics)

        print(f"{model_name} last-cycle metrics:")
        print(last_metrics)

    # -----------------------------------------------------
    # 7) SONUÇLARI TABLOYA ÇEVİR
    # -----------------------------------------------------

    all_cycle_df = pd.DataFrame(all_cycle_results)
    all_cycle_df = all_cycle_df[
        ["dataset", "model", "evaluation", "MAE", "RMSE", "R2"]
    ]

    last_cycle_df = pd.DataFrame(last_cycle_results)
    last_cycle_df = last_cycle_df[
        ["dataset", "model", "evaluation", "MAE", "RMSE", "R2"]
    ]

    # -----------------------------------------------------
    # 8) CSV OLARAK KAYDET
    # -----------------------------------------------------

    all_cycle_df.to_csv(
        dataset_metrics_dir / f"{dataset_id}_baseline_metrics_all_cycles.csv",
        index=False,
    )

    last_cycle_df.to_csv(
        dataset_metrics_dir / f"{dataset_id}_baseline_metrics_last_cycle.csv",
        index=False,
    )

    # -----------------------------------------------------
    # 9) MODEL KARŞILAŞTIRMA GRAFİKLERİNİ KAYDET
    # -----------------------------------------------------

    plot_model_comparison(
        all_cycle_df,
        "RMSE",
        dataset_figures_dir / "model_comparison_rmse_all_cycles.png",
    )

    plot_model_comparison(
        last_cycle_df,
        "RMSE",
        dataset_figures_dir / "model_comparison_rmse_last_cycle.png",
    )

    print(f"\n{dataset_id} tamamlandı.")

    print("\nAll-cycle evaluation:")
    print(all_cycle_df)

    print("\nLast-cycle evaluation:")
    print(last_cycle_df)

    return all_cycle_df, last_cycle_df


# ---------------------------------------------------------
# ANA PROGRAM
# ---------------------------------------------------------
# Bütün dataset'leri sırayla çalıştır.
#
# FD001 -> çalıştır
# FD002 -> çalıştır
# FD003 -> çalıştır
# FD004 -> çalıştır
#
# Her dataset'in sonuçlarını bir listede topla.
# En sonda hepsini tek summary CSV dosyasına yaz.
# ---------------------------------------------------------


def main():
    all_datasets_all_cycle = []
    all_datasets_last_cycle = []

    # DATASETS listesindeki her dataset için aynı pipeline'ı çalıştır.
    for dataset_id in DATASETS:
        all_cycle_df, last_cycle_df = run_one_dataset(dataset_id)

        all_datasets_all_cycle.append(all_cycle_df)
        all_datasets_last_cycle.append(last_cycle_df)

    # Her dataset'ten gelen all-cycle tablolarını alt alta birleştir.
    summary_all_cycle_df = pd.concat(
        all_datasets_all_cycle,
        ignore_index=True,
    )

    # Her dataset'ten gelen last-cycle tablolarını alt alta birleştir.
    summary_last_cycle_df = pd.concat(
        all_datasets_last_cycle,
        ignore_index=True,
    )

    # Genel metrics klasörünü garantiye al.
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # Tüm dataset'lerin özet all-cycle sonuçlarını kaydet.
    summary_all_cycle_df.to_csv(
        METRICS_DIR / "summary_baseline_metrics_all_cycles.csv",
        index=False,
    )

    # Tüm dataset'lerin özet last-cycle sonuçlarını kaydet.
    summary_last_cycle_df.to_csv(
        METRICS_DIR / "summary_baseline_metrics_last_cycle.csv",
        index=False,
    )

    print("\n==============================")
    print("TÜM BASELINE PIPELINE TAMAMLANDI")
    print("==============================")

    print("\nSummary - All-cycle evaluation:")
    print(summary_all_cycle_df)

    print("\nSummary - Last-cycle evaluation:")
    print(summary_last_cycle_df)


# ---------------------------------------------------------
# PROGRAMI BAŞLAT
# ---------------------------------------------------------
# Bu dosya doğrudan çalıştırılırsa main() fonksiyonunu çağır.
#
# Örnek:
# python src/train_baseline_all_datasets.py
# ---------------------------------------------------------

if __name__ == "__main__":
    main()