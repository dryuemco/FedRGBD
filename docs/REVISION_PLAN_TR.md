# NCAA-D-26-02211 — Revizyon Planı (Türkçe çalışma notu)

Karar: **Major Revision**, teslim tarihi **13 Kasım 2026** (Editör: Ö. F. Ertuğrul).
Kod tarafı `revision-ncaa` dalında tamamlandı (bkz. `docs/REVISION_CHANGES.md`, İngilizce ve
dosya dosya). Bu belge üç soruya cevap verir:

1. Hakemler ne istedi, biz **neyi ne amaçla** yaptık?
2. Cihazlara (3× Jetson) geçmeden önce başka ne yapılmalı?
3. Düzeltmeler için öncelikli deney ve metin planı nedir?

---

## 1. Hakem yorumu → yapılan iş → amaç

| # | Hakem isteği | Yapılan (kod) | Amaç / makalede nereye gidecek |
|---|--------------|---------------|--------------------------------|
| R3-a | FLAME'de aynı video/sekanstan kareler hem train hem test'te olabilir; sekans/kaynak düzeyinde bölme yapın | `scripts/analyze_flame_leakage.py` (algısal hash ile yakın-kopya kümeleme + mevcut `data/processed` için sızıntı raporu) ve `data_splitter.py --group_file` (yakın-kopya grubu bir bütün olarak tek node'a ve tek split'e gider) | Metodoloji bölümüne "sekans düzeyinde (grup) bölme" paragrafı; sızıntı oranını **rapor edip** v1 sonuçlarıyla farkı tartışmak. Sonuçlar 99 %'un altına inerse bu, hakemin "doygunluk" itirazına da cevap olur |
| R3-b, R5-b | Yalnızca accuracy yetmez: balanced accuracy, macro-F1, sensitivity, specificity, precision, recall, MCC, ROC-AUC; global **ve** istemci başına; istemci başına karışıklık matrisi | `src/evaluation/metrics.py`; istemci her turda tüm metrikleri döndürüyor; sunucu `results.json`'a istemci başına satır + ağırlıklı global + havuzlanmış (toplam karışıklık matrisinden) metrikleri yazıyor; `train_local.py` / `train_centralized.py` aynı seti epoch başına ve test'te kaydediyor | Yeni tablo: her strateji × dağılım için tüm metrikler (mean ± std, %95 GA); ek tablo: Non-IID'de istemci başına karışıklık matrisleri |
| R3-c | Yöntemleri tur sayısına değil **geçen süreye ve iletişim maliyetine** göre de karşılaştırın | İstemci `fit_time_s`, `eval_time_s`, gönderilen/alınan bayt; sunucu tur sonunda `elapsed_s` ve kümülatif iletişim baytı; `analyze_results.py` accuracy-vs-time ve accuracy-vs-MB grafikleri | Yeni şekil (3 panel: tur / süre / MB). Eski sonuçlar için süre-MB **tahmini** (grafikte kesikli çizgi ve dipnot) |
| R3-d, R5-a | 3 seed az; güven aralığı ekleyin; "istatistiksel olarak ayırt edilemez" gibi ifadelerden kaçının | Matriste 5 seed bloğu; `analyze_results.py`: mean ± std, t-dağılımıyla %95 GA, eşleştirilmiş Cohen d, Wilcoxon, eşleştirilmiş t, Friedman | İstatistik paragrafını yeniden yazmak (GA ile), "indistinguishable" yerine "no significant difference at n=5 (p=…)" |
| R3-e | Daha zor bir koşul: daha az yerel örnek, daha güçlü heterojenlik | `--subsample_frac 0.05 0.01` (node başına train'i %5 / %1'e indirir) ve `--dirichlet_alpha 0.1 0.5 1.0` | Yeni alt bölüm "Low-data regime" ve "Dirichlet label skew"; FL'nin yerel eğitime göre ne zaman avantaj sağladığını burada göstermek |
| R5-c | Sonuçlar farklı non-IID derece ve türlerinde stabil mi? Tek elle kurgulanmış label-skew yetmez | Dirichlet bloğu (α = 0.1, 0.5, 1.0) | Aynı alt bölüm; α'ya göre FedAvg/FedProx eğrileri |
| R5-d | Hiperparametre duyarlılığı: μ yalnızca 0.01/0.1; batch 8, Adam 1e-3, 5 yerel epoch sabit | Matriste μ ∈ {0.001, 0.01, 0.05, 0.1, 0.5}, E ∈ {1, 2, 5}, lr ∈ {1e-4, 1e-3} blokları | "Sensitivity analysis" alt bölümü + 1 şekil |
| R3-f, R5-e | FedBN sonuçları 3 turla sınırlı; "çok turlu senaryoya saklayın" iddiası test edilmedi | 10 turluk FedBN (+ referans FedAvg) bloğu | FedBN paragrafını 10 tur sonucuna göre yeniden yazmak; iddiayı test edilen koşulla sınırlamak |
| R5-f | Başlık değişsin: "Empirical Evaluation of Federated Learning on Edge GPU Clusters with Heterogeneous RGB-D Sensors" | — (metin işi) | Başlığı hakemin önerdiği gibi değiştirmek |
| R1 | Eş. 1–5b öncesine ilgili referanslar; 2025-26 kaynakları; gelecek çalışmaya wavelet tabanlı hibrit yöntem | — (metin işi) | Related work + future work |
| R5-g | Metasezgisel hiperparametre ayarlama literatürüne atıf ("Convolutional neural networks hyperparameters tuning") | — (metin işi) | Related work'e 1 paragraf |
| R3-g | Çapraz-sensör deneyi sahne bağımsız olmalı (leave-one-scene-out); sensör heterojenliği FL deneyine bağlansın | `scripts/cross_sensor_loso.py` + `src/data/custom_dataset.py` (sahne başına bir kat, rastgele-bölme taban çizgisiyle yan yana); sahne etiketleri için `labels.csv` gerekir | Yeni tablo: LOSO vs rastgele bölme, kamera içi / kameralar arası; veri Jetson'larda, koşu cihaz ister (bkz. §3.4) |

### Bu oturumda yapılan destekleyici işler
* `configs/experiment_matrix.yaml` → `revision:` bloğu ve `scripts/print_revision_commands.py`
  (komutları üretir, `results/<run>/results.json` varsa atlar; **kaldığı yerden devam edilebilir**).
* `results.json` şeması v2: eski anahtarlar aynen korunuyor, yeni `rounds`, `client_config`,
  `model_payload_bytes`, `tags` vb. eklendi. Eski dosyalar ve yeni dosyalar aynı analiz
  betiğinden geçiyor.
* 220 CPU birim testi (`python3 -m pytest tests -q`), gerçek Flower 1.13.1 ile localhost'ta
  2 istemcili uçtan uca test dahil.
* `results/` altına hiçbir şey yazılmadı; model değişmedi.

---

## 2. Cihazlara geçmeden önce kontrol listesi

Sıra önemli; 2.1–2.4 tamamen CPU işidir ve masaüstünde yapılabilir.

### 2.1 Sızıntı analizini **gerçek** FLAME verisi üzerinde çalıştır (kararı bu belirler)
```bash
python3 scripts/analyze_flame_leakage.py --data_dir data/raw/flame_dataset \
    --processed_dir data/processed --output_dir analysis/leakage --threshold 8 --workers 4 \
    --sweep 4 6 8 10 12 --sequence_heuristic --examples 20
```
`--sweep` eşik duyarlılığı tablosunu (`threshold_sweep`) `groups.json`'u
değiştirmeden ekler; `--sequence_heuristic` dosya adlarındaki ardışık kare
numaralarının ne kadarının aynı yakın-kopya grubuna düştüğünü raporlar;
`--examples` en büyük grupları `example_groups.txt`'ye yazar;
`--hash both --phash_threshold 10` dHash ve pHash kenarlarının **birleşimini**
kümeler (daha temkinli, daha kaba gruplama); bayt düzeyinde birebir kopyalar
`exact_duplicate_files_md5` olarak raporlanır.
`analysis/leakage/leakage_report.json` içindeki `global_test_leak_rate_any_node` ve
`groups_spanning_multiple_nodes` sayılarına bak. Beklenti: FLAME video karesi olduğu için
oran yüksek çıkar. **Eşik seçimi** (`--threshold 6/8/10`) için `group_size_histogram`'ı ve
`cross_label_groups`'u kontrol et; cross-label grup çoksa eşik fazla gevşektir. Bu sayılar
makaleye girecek (R3-a'ya doğrudan cevap).

### 2.2 Protokol kararı: yeniden bölme yapılacak mı?
Sızıntı anlamlıysa (büyük olasılıkla evet) tüm revizyon deneyleri **grup-güvenli bölme** ile
koşulmalı. Sonuçları:
* Eski 3 seed'lik sonuçlar yeni koşularla **karşılaştırılamaz** (farklı bölme). Seed
  uzatma bloğu 2 yeni seed değil **5 seed'in tamamı** olarak koşulur:
  `print_revision_commands.py --all_seeds`. Makalede v1 sonuçları "image-level split"
  olarak ayrı bir tabloda kalır; ana tablolar yeni protokolden gelir.
* Dizin adı `results/rev_*` olduğu için eski `results/3node_*` ile karışmaz.

Sızıntı ihmal edilebilir düzeydeyse (örn. < %2) eski seed'ler yeniden kullanılabilir ve
yalnızca 789/1011 koşulur; bu durumda makalede sızıntı oranını raporlamak yeterlidir.

### 2.3 Bölmeleri üret ve **üç node'da aynı olduğunu doğrula**
```bash
python3 src/data/data_splitter.py --data_dir data/raw/flame_dataset --output_dir data/processed \
    --nodes 3 --seed 42 --group_file analysis/leakage/groups.json \
    --dirichlet_alpha 0.1 0.5 1.0 --subsample_frac 0.05 0.01 --clean --verify
```
`--clean` **zorunludur**: `data/processed` zaten bir bölme içerdiği için bu komut
yeniden bölme yapar. Bölücü yalnızca *aynı* dosya adını üzerine yazar; `--clean`
olmadan başka bir node'a veya başka bir train/val/test kovasına taşınan
görüntüler eski bölmeden kalır ve ağaç iki bölmeyi birden tutar (eğitim
görüntüleri sessizce val/test'e sızar). `--verify` yazma bittikten sonra
`manifest.csv` dosyalarını yeniden okur; hiçbir `(group_id,label)` biriminin
node'lara veya train/val/test'e bölünmediğini ve manifest sayımlarının
`split_stats.json` ile eşleştiğini doğrular, `VERIFY PASS`/`VERIFY FAIL` basar ve
FAIL durumunda 1 koduyla çıkar.
Dikkat: bölücü `os.walk` sırasına dayanır (v1'de de öyleydi). Her node kendi kopyasında
çalıştırıyorsa dosya sistemi sırası farklıysa bölmeler **farklı** çıkabilir ve node'lar
arası örtüşme oluşur. Doğrulama: üç node'da
`md5sum data/processed/*/manifest.csv data/processed/split_stats.json` çıktıları birebir
aynı olmalı. Aynı değilse tek node'da üretip `data/processed`'ı (sembolik linkler, küçük)
diğerlerine kopyala ya da `manifest.csv`'yi paylaş.
`split_stats.json` → `class_counts` tabloları makaledeki "veri dağılımı" tablosuna girer.

**Grup modunda birimler devasa (17 Eylül 2026 gerçek veri bulgusu):** 47.992 görüntünün
47.863'ü 265 gruba düşüyor; en büyük birim tek sınıftan 4.341 görüntü (bir video). İlk gerçek
koşuda IID bölmesinde node_b'nin val kümesi **boş** kaldı, Dirichlet(0.1) bir node'a 12 görüntü
verdi, "%5" alt örnek bir node'da %69 çıktı. Bu yüzden bölücü grup modunda artık (a) aynı
görüntü kotalarını **en-büyük-önce greedy** ile dolduruyor (yazarların kendi "en büyük açık"
kuralı), (b) alt örneklemeyi node'un kendi birimleri **içinde görüntü düzeyinde** yapıyor
(kesin oran; düşen görüntü hiçbir yere gitmediği için sızıntı garantisi bozulmaz; makalede
"kare sayısı azalır, sahne sayısı değil" denmeli), (c) `--verify` öncesi taze
`split_stats.json` yazıyor (eskiden diskteki bayat dosyayla karşılaştırıp yanlış FAIL
veriyordu). Varsayılan (grupsuz) yol bit-birebir aynı. Dirichlet için `--dirichlet_min_size
200` kullanıldı: 10 ile node 12 görüntüde kalıyor ve val/test ölçülemiyor; 200 ≈ node başına
30 val + 30 test görüntüsünün alt sınırı. Gerçekleşen sayılar `split_stats.json` →
`class_counts`'ta; hedef 70/15/15 ve eşit node büyüklüğünden sapmalar (en büyük birim
kadar) makalede açıkça raporlanacak.

### 2.4 Yeni bölmelerin sızıntısını doğrula (sıfır olmalı)
```bash
python3 scripts/analyze_flame_leakage.py --data_dir data/raw/flame_dataset \
    --processed_dir data/processed --output_dir analysis/leakage
```
Tüm `leak_rate_any_node` = 0 ve `groups_spanning_multiple_nodes` = 0 olmalı.

### 2.5 Kodu üç Jetson'a al ve doğrula
```bash
git fetch && git checkout revision-ncaa          # her node'da
pip install pytest                                # yalnızca test için
python3 -m pytest tests -q -k "not end_to_end"    # ~1 dk, CPU
python3 -m pytest tests/test_fl_end_to_end.py -q  # isteğe bağlı, ~1-2 dk
```
Yeni Python bağımlılığı yok (numpy 1.26.4 / torch 2.5 / flwr 1.13.1 aynen).

### 2.6 Donanımda 1 turluk duman testi (matris öncesi)
`iid_sub0.01` bölmesi (node başına ~110 eğitim görüntüsü) ile 1 tur FedAvg koş ve
`results/smoke/results.json` içinde `rounds[0].evaluate.clients.node_*` altında tüm
metriklerin, `payload_bytes_up` ≈ 6.13 MB ve `elapsed_s` değerlerinin geldiğini kontrol et:
```bash
# Node A
python3 src/fl/server.py --strategy fedavg --rounds 1 --seed 42 --min_clients 3 --output_dir results/smoke --tag smoke
# her node
python3 src/fl/client.py --server 192.168.1.4:8080 --data_dir data/processed/iid_sub0.01/node_X --batch_size 8 --seed 42
python3 scripts/analyze_results.py --results_dir results --output_dir analysis --include_test_runs
```
Sonra `results/smoke` silinir (matris dizin adlarıyla çakışmaz ama temiz kalsın).

### 2.7 Küçük kararlar (şimdi netleştir)
* **Değerlendirme kümesi**: v1'de olduğu gibi her tur *val* üzerinde ölçüm yapılıyor.
  Hakemler test-kümesi rakamı da isteyebilir; seçenekler: (a) v1 ile aynı kal, metinde
  açıkça söyle; (b) son turdan sonra istemcileri `--eval_split test` ile bir kez daha
  bağlayıp 1 turluk "değerlendirme koşusu" yap. Öneri: (a) + subsample/Dirichlet için de aynı
  kural; gerekirse hakem cevabında (b)'yi ek olarak sun.
* **Eşik**: `--threshold 8` varsayılan; 2.1'deki histogramla teyit et ve makaleye yaz.
* **Dirichlet `--dirichlet_min_size`**: varsayılan 10; α=0.1'de bir node'un çok küçük
  kalması normaldir, bu **istenen** zorluktur. Bunu kaldırma, sadece rapor et.
* **`.eml` dosyası** git'e girmiyor (`*.eml` ignore'da); repo public olduğu için hakem
  metinlerini asla commit etme.

---

## 3. Deney planı (öncelikli, süre tahminli)

Süreler mevcut `results/` dosyalarındaki ölçülmüş `total_time_s`'den türetildi
(3 tur, 5 yerel epoch): FedAvg ≈ 1.7 sa, FedProx ≈ 2.9 sa, FedBN ≈ 1.75 sa,
centralized ≈ 4.2 sa, local-only (3 node ardışık) ≈ 4.4 sa. Subsample koşuları eğitim
verisiyle orantılı kısalır (%5 → ~0.3 sa, %1 → ~0.2 sa; kaba tahmin). 10 tur ≈ 3.3× 3 tur.
Testbed tek seferde yalnızca bir FL koşusu yapabilir.

| Öncelik | Blok | Koşu | Tahmini testbed süresi | Hangi hakem isteği |
|---------|------|------|------------------------|--------------------|
| **P0** | 2.1–2.4 sızıntı + yeniden bölme | CPU | birkaç saat, cihaz gerekmez | R3-a |
| **P0** | `seed_extension` (**5 seed**, FedAvg + FedProx 0.01, IID + Non-IID) | 20 | ≈ 46 sa | R3-d, R5-a, R3-b (yeni metrikler bu koşulardan gelir) |
| **P0** | `low_data` (%5, %1 × IID/Non-IID × 2 strateji × 3 seed) | 24 | ≈ 8 sa | R3-e |
| **P0** | `mu_grid` (0.001, 0.05, 0.1, 0.5; 0.01 P0'dan paylaşılır) | 12 | ≈ 35 sa | R5-d |
| **P1** | `dirichlet_skew` (α 0.1, 1.0 önce; 0.5 sonra) | 18 | ≈ 41 sa | R3-e, R5-c |
| **P1** | `long_horizon_fedbn` (10 tur FedBN + FedAvg, 3 seed) | 6 | ≈ 35 sa | R3-f, R5-e |
| **P1** | `local_epochs` (E=1, 2) | 12 | ≈ 12 sa | R5-d |
| **P1** | `learning_rate` (1e-4) | 6 | ≈ 14 sa | R5-d |
| **P2** | `baselines_extension` (centralized + local: yeni seed'ler, Dirichlet, subsample) | 50 | ≈ 120 sa testbed **veya** masaüstü GPU'da | R3-e, R5 (alt/üst sınır) |

Toplam ≈ 310 testbed saati (~13 gün kesintisiz). Öneriler:
* P0 (~90 sa) bitince makalenin ana tabloları/şekilleri üretilebilir; P1 paralel yazım
  sırasında koşar.
* Baseline'lar (P2) FL ağı gerektirmez: centralized/local koşularını masaüstü GPU'da
  yap, makalede "accuracy baselines were trained on a desktop GPU; timing not comparable"
  de. Böylece testbed FL'ye kalır. Ya da FL koşusu gece testbed'deyken gündüz tek node'da
  koş (node B/C boşken).
* Her koşudan sonra `python3 scripts/analyze_results.py` ile tabloları güncelle; hiçbir
  şeyi elle tabloya yazma (rakamlar `analysis/summary_table.md`'den gelir).
* Komutlar: `python3 scripts/print_revision_commands.py --all_seeds` (tüm liste) veya
  `--block mu_grid --format bash` (o bloğun sunucu betiği).

### 3.4 Kod olarak hâlâ eksik olan (cihaz + özel veri gerektirir)
* **Leave-one-scene-out çapraz-sensör değerlendirmesi** (R3-g): özel RGB-D kayıtlarındaki
  5 sahne için "4 sahnede eğit, 1'inde test" döngüsü. Kayıtlar Jetson'larda; küçük bir betik
  (`scripts/cross_sensor_loso.py`) gerekir. İstersen bir sonraki adımda yazabilirim; veri
  klasör yapısını (`data/raw/custom/node_*/{id}_rgb.png` + sahne etiketi nerede?) bilmem
  gerekir.
* Sensör heterojenliğini FL deneyine bağlamak (R3): en ucuz yol, custom RGB(-D) kayıtlarını
  node başına doğal istemci verisi olarak kullanan küçük bir FL koşusu (her node kendi
  kamerasının verisiyle). Bu da yeni veri/etiket gerektirir; kapsam kararı sana ait.

---

## 4. Metin planı (kod gerektirmez, deneyler koşarken yapılabilir)
1. **Başlık**: R5'in önerdiği "Empirical Evaluation of Federated Learning on Edge GPU
   Clusters with Heterogeneous RGB-D Sensors".
2. **Kapsam ifadesi**: giriş ve sonuçta "3 node, 1 veri kümesi, 1 backbone, 3 tur" sınırını
   açıkça yaz; genellenebilir olan/olmayan sonuçları ayır (R5).
3. **Veri/Protokol**: sekans düzeyinde bölme, sızıntı oranı (2.1'den), Dirichlet ve
   low-data tanımları, metrik tanımları (specificity, MCC, AUC formülleri).
4. **İstatistik**: n=5 ile GA'lı tablolar; "indistinguishable" ifadelerini kaldır; Cohen d'yi
   "large but n is small" diye yorumla; Friedman/Wilcoxon p değerlerini ver.
5. **FedBN**: 10 tur sonucuna göre yeniden yaz; iddiayı test edilen koşulla sınırla.
6. **Süre/iletişim**: FedProx'un ilk tur avantajını süre eksenindeki grafikle birlikte
   tartış ("aynı duvar-saati bütçesinde FedAvg 2. turu bitiriyor" gibi).
7. **R1**: Eş. 1–5b öncesine FedAvg/FedProx/FedBN orijinal atıfları; 2025-26 kaynakları;
   future work'e wavelet tabanlı hibrit yaklaşım cümlesi.
8. **R5-g**: metasezgisel hiperparametre ayarlama literatürüne 1 paragraf (hakemin verdiği
   çalışma + 1-2 genel kaynak).
9. **Cevap mektubu**: yukarıdaki tablonun (§1) hakem-madde sırasına göre düzenlenmiş hâli;
   her madde için "değişiklik nerede (bölüm/tablo/şekil)" satırı.

---

## 5. Jetson gerektirmeyen işler — durum (17 Eylül 2026)

Bu bölüm, cihazlar gelmeden masaüstünde yapılabilecek her şeyin envanteri ve durumudur.
Kod ortamı: `C:\Users\CORSAIR\venvs\fedrgbd` (CPU, testler) ve `C:\Users\CORSAIR\venvs\fedrgbd-gpu`
(RTX 5090, baseline koşuları; bkz. `docs/DESKTOP_GPU_BASELINES.md`). Sistem Python'unda ve
WSL'de torch **yok**; komutları bu venv'lerle çalıştır.

| # | İş | Durum | Nerede |
|---|----|-------|--------|
| 1 | Kaynakça: 15 `% VERIFY` kaydı yayıncı/Crossref/arXiv kaydına karşı doğrulandı; 3'ü arXiv → yayımlanmış sürüme düzeltildi (Banerjee → Euro-Par 2025 LNCS 15900; Zhang → ICASSP 2025; Borazjani → IEEE TAI 7(9)). Hiçbiri başarısız olmadı | **bitti** | `paper/main.tex` kaynakça, `docs/RESPONSE_TO_REVIEWERS.md` R1.2 |
| 2 | v1'den kalan boş bölümler (Modality Ablation, Resource Profiling, Network Constraint Sensitivity) kaldırıldı; enerji/latency vaatleri metinden çıkarıldı; üç ölçüm Limitations + Future Work'te "kapsam dışı" olarak yazıldı | **bitti** | `paper/main.tex` §Limitations, §Future Work; cevap mektubu R3.1 |
| 3 | v1 (image-level) sonuçları yeni analiz boru hattından geçirildi; `tab:v1_ci`, `tab:time`, `tab:stats` hücreleri birebir doğrulandı; tek tutarsızlık (IID'de p<0.05 çift sayısı 3 değil 2) düzeltildi | **bitti** | `analysis/` (CSV/MD/grafikler), `paper/tables/*.tex` |
| 4 | Masaüstü GPU ortamı: torch 2.11+cu128, CUDA doğrulandı, 227 test geçti, sentetik veride `Device: cuda` duman testi; `results.json`'a `device` alanı eklendi | **bitti** | `docs/DESKTOP_GPU_BASELINES.md`, `setup_desktop_windows.ps1` |
| 5 | P0 zinciri tek komut: v1 sızıntı denetimi → ham veri denetimi (sweep, sekans, MD5) → karar özeti → `--clean --verify` yeniden bölme → sıfır-sızıntı denetimi + manifest md5'leri | **bitti (17 Eylül, gerçek veri)** | `scripts/run_p0_leakage_and_split.py`, `analysis/leakage/P0_SUMMARY.md` |
| 6 | FLAME ham verisi (Kaggle `archive.zip`, 47.992 jpg) masaüstüne indirildi, `data/raw/flame_dataset/{Fire,No_Fire}` olarak düzleştirildi; yeni `data/processed` 15 bölme, hepsi sızıntısız | **bitti** | `data/processed/split_stats.json`, §5.1 |
| 7 | P2 baseline'lar (centralized + local, **62 koşu**: iid/skew 5 seed, Dirichlet ve low-data 3 seed) masaüstü GPU'da | **bitti (17 Eylül 12:15–20:23, 4 paralel şerit)** | `results/rev_*_{centralized,local}_seed*`, §5.2 |
| 8 | Fon numarası / AI-disclosure ifadesi; testbed fotoğrafı | yazar girdisi | `paper/main.tex` `\todo` |

Kalan 27 `\todo` yer tutucusunun hepsi Jetson koşularına veya cihazdaki özel veriye (sahne
etiketleri, LOSO) bağlıdır. Sızıntı tablosu (`tab:leakage`) ve gerçekleşen bölme sayıları
tablosu (`tab:group_counts`) gerçek sayılarla dolduruldu. Cihazlar gelmeden yapılabilecek tek
iş 7. satır (masaüstü GPU baseline'ları).

### 5.1 P0 bulguları (17 Eylül 2026, gerçek FLAME verisi)

| Bulgu | Değer | Nereye gitti |
|---|---|---|
| v1 image-level bölmede val/test görüntülerinin bir node'un train'inde yakın-kopyası olan payı | IID %99,64 / %99,65; label-skew %99,58 / %99,65 | `tab:leakage` (c), §IV-B metni, cevap mektubu R3.2 |
| Node'lara yayılan grup sayısı (v1) | 235 (IID) / 219 (skew) | aynı |
| Grup istatistiği (τ=8) | 394 grup, 265'i >1 görüntü, 47.863 görüntü (%99,73); en büyük grup 4.924 | `tab:leakage` (a) |
| Eşik taraması | τ=4: 2.051 grup; τ=8: 394; τ=10: en büyük bileşen 34.337 (%72) → τ=8 seçimi gerekçesi | `tab:leakage` (b), §III-D |
| Ardışık kare çiftlerinin aynı gruba düşme oranı | %99,06 | §III-D |
| **Çapraz-etiketli gruplar** | 22 grup, 20.006 görüntü (%42); en büyüğü 4.341 Fire + 583 No_Fire | §III-D: grup etiketten bağımsız bütün tutulur |
| Yeni bölmelerin denetimi | 15 bölmenin hepsinde L = 0, node'lara yayılan grup 0 | `tab:leakage` (c), `analysis/leakage/post_split_audit/` |
| Gerçekleşen node sayıları | IID 11.139–11.346 train / 2.326–2.519 val-test; Dirichlet(0.1) node A %9 ateş, B %95; Dirichlet(0.5) en küçük istemci 911 görüntü | `tab:group_counts` |

**Bölücüde üç düzeltme gerekti** (hepsi yalnızca grup modunda, varsayılan yol bit-birebir):
1. `--verify` bayat `split_stats.json` ile karşılaştırıyordu → taze dosya doğrulamadan önce yazılıyor.
2. Kümülatif kesme dev birimlerle boş kova bırakıyordu (node_b val = 0) → en-büyük-önce greedy.
3. Çapraz-etiketli grubun Fire ve No_Fire parçaları ayrı birimlerdi → iki node'a / train-test'e dağılıyor,
   bağımsız denetim %72'ye kadar sızıntı buluyordu → grup **etiketten bağımsız tek paket**; iki
   doğrulayıcı (`scripts/verify_splits.py`, bölücü içi) artık etiketten bağımsız da kontrol ediyor.
Alt örnekleme grup modunda node'un kendi grupları içinde **kare düzeyinde** (kesin %5 / %1);
makalede "kare sayısı azalır, sahne sayısı değil" dendi. Dirichlet için `--dirichlet_min_size 200`.

### 5.2 Baseline bulguları (grup düzeyinde bölme, masaüstü GPU)

| Koşul | Centralized | Local-only (node ortalaması) | Not |
|---|---|---|---|
| IID (n=5) | acc 90,9 ± 2,4; bal 92,0 | acc 78,3 ± 5,0; bal 76,6 | v1'de 99,6 / 99,5 idi; 12,6 puanlık boşluk açıldı |
| Label-skew (n=5) | acc 94,2; bal 94,5; MCC 0,88 | acc 94,1; bal 87,9; MCC 0,78 | accuracy eşit, balanced/MCC değil → hakemin metrik itirazı kendi verimizde görünüyor |
| Dirichlet α=0,1 (n=3) | bal 90,5 | **bal 58,7** (acc 90,7) | iki node hiç no-fire görmüyor → tek sınıf tahmini |
| Dirichlet α=0,5 / 1,0 | bal 88,8 / 76,0 | bal 80,1 / 87,7 | centralized'ın 90→76 düşüşü test sekans bileşiminden |
| Low-data ρ=0,05 / 0,01 (IID) | bal 89,7 / 89,1 | bal 81,0 / 80,4 | kare düzeyinde alt örnek sekansları koruyor; local az kaybediyor |
| Low-data ρ=0,05 / 0,01 (skew) | bal 91,6 / 90,8 | bal 85,3 / 79,8 | |

Makalede dolduruldu: `tab:protocol_effect` (Centralized/Local-only satırları iki protokolde),
`tab:fullmetrics`, `tab:dirichlet`, `tab:lowdata` baseline satırları; IV-C/IV-E/IV-F metinleri;
Limitations'a "sonuçlar tutulan sekans kümesine koşullu" paragrafı; cevap mektubu R3.4 sonrası.
FL satırları (`\PHs`) Jetson koşularını bekliyor. `analyze_results.py` artık `{group}` /
`{image}` protokolünü ayrı tutuyor (v1 ile yeni koşular asla aynı hücrede toplanmaz);
`print_revision_commands.py --all_seeds` baseline bloğunu da 5 seed'e çıkarıyor.

**Jetson'lara taşınacak:** `data/processed` bu makinede hardlink ile üretildi (1,6 GB). Üç node'a
ya bu ağaç kopyalanır ya da her node'da `run_p0_leakage_and_split.py --skip_download` koşulup
`P0_SUMMARY.md`'deki manifest md5'leri karşılaştırılır (bölücü `os.walk` sırasına bağlıdır;
md5'ler tutmuyorsa tek kopyayı dağıt).
