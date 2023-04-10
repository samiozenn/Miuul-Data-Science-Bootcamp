#####################################################
# AB Testi ile BiddingYöntemlerinin Dönüşümünün Karşılaştırılması
#####################################################
import pandas as pd

#####################################################
# İş Problemi
#####################################################

# Facebook kısa süre önce mevcut "maximumbidding" adı verilen teklif verme türüne alternatif
# olarak yeni bir teklif türü olan "average bidding"’i tanıttı. Müşterilerimizden biri olan bombabomba.com,
# bu yeni özelliği test etmeye karar verdi veaveragebidding'in maximumbidding'den daha fazla dönüşüm
# getirip getirmediğini anlamak için bir A/B testi yapmak istiyor.A/B testi 1 aydır devam ediyor ve
# bombabomba.com şimdi sizden bu A/B testinin sonuçlarını analiz etmenizi bekliyor.Bombabomba.com için
# nihai başarı ölçütü Purchase'dır. Bu nedenle, istatistiksel testler için Purchasemetriğine odaklanılmalıdır.




#####################################################
# Veri Seti Hikayesi
#####################################################

# Bir firmanın web site bilgilerini içeren bu veri setinde kullanıcıların gördükleri ve tıkladıkları
# reklam sayıları gibi bilgilerin yanı sıra buradan gelen kazanç bilgileri yer almaktadır.Kontrol ve Test
# grubu olmak üzere iki ayrı veri seti vardır. Bu veri setleriab_testing.xlsxexcel’ininayrı sayfalarında yer
# almaktadır. Kontrol grubuna Maximum Bidding, test grubuna AverageBiddinguygulanmıştır.

# impression: Reklam görüntüleme sayısı
# Click: Görüntülenen reklama tıklama sayısı
# Purchase: Tıklanan reklamlar sonrası satın alınan ürün sayısı
# Earning: Satın alınan ürünler sonrası elde edilen kazanç



#####################################################
# Proje Görevleri
#####################################################

######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################

# 1. Hipotezleri Kur
# 2. Varsayım Kontrolü
#   - 1. Normallik Varsayımı (shapiro)
#   - 2. Varyans Homojenliği (levene)
# 3. Hipotezin Uygulanması
#   - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi
#   - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi
# 4. p-value değerine göre sonuçları yorumla
# Not:
# - Normallik sağlanmıyorsa direkt 2 numara. Varyans homojenliği sağlanmıyorsa 1 numaraya arguman girilir.
# - Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.




#####################################################
# Görev 1:  Veriyi Hazırlama ve Analiz Etme
#####################################################

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option("display.width", 500)

# Adım 1:  ab_testing_data.xlsx adlı kontrol ve test grubu verilerinden oluşan veri setini okutunuz. Kontrol ve test grubu verilerini ayrı değişkenlere atayınız.
cont_df = pd.read_excel("4. Hafta Ölçüm Problemleri/Ödevler/ABTesti-221114-234653/ABTesti/ab_testing.xlsx", "Control Group")
test_df = pd.read_excel("4. Hafta Ölçüm Problemleri/Ödevler/ABTesti-221114-234653/ABTesti/ab_testing.xlsx", "Test Group")
# Adım 2: Kontrol ve test grubu verilerini analiz ediniz.

cont_df.describe().T
cont_df.info()
test_df.describe().T
test_df.info()
# Adım 3: Analiz işleminden sonra concat metodunu kullanarak kontrol ve test grubu verilerini birleştiriniz.

df_ = pd.concat([cont_df, test_df], axis=0, ignore_index=True)

#####################################################
# Görev 2:  A/B Testinin Hipotezinin Tanımlanması
#####################################################

# Adım 1: Hipotezi tanımlayınız.

# H0: M1 = M2 (İki ortalama arasında istatistiksel olarak anlam ifade eden bir fark yoktur.)
# H1: M1 != M2 (Fark vardır.)


# Adım 2: Kontrol ve test grubu için purchase(kazanç) ortalamalarını analiz ediniz

test_df["Purchase"].mean()
cont_df["Purchase"].mean()

# cont_df.groupby("Purchase").mean().reset_index()
#####################################################
# GÖREV 3: Hipotez Testinin Gerçekleştirilmesi
#####################################################

######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################


# Adım 1: Hipotez testi yapılmadan önce varsayım kontrollerini yapınız.Bunlar Normallik Varsayımı ve Varyans Homojenliğidir.

# Kontrol ve test grubunun normallik varsayımına uyup uymadığını Purchase değişkeni üzerinden ayrı ayrı test ediniz

# Normallik varsayımı:
# H0: Dağılım normaldir.        p-value < 0.05 ise RED
# H1: Dağılım normal değildir.  p-value > 0.05 ise RED

def normal_var(col_name, test=False):
    if test:
       shap_stat, pvalue = shapiro(test_df[col_name])
       print(pvalue)
    else:
       shap_stat, pvalue = shapiro(cont_df[col_name])
       print(pvalue)

normal_var("Purchase")
normal_var("Purchase", test=True)

# H1: Dağılım normal değildir.  p-value > 0.05 ise RED


# Varyans homojenliği varsayımı:
# H0: Varyanslar homojen dağılmıştır.            p-value < 0.05 ise RED
# H1: Varyanslar homojen dağılmamıştır.          p-value > 0.05 ise RED

def var_homo(col_name_1, col_name_2):
    var_stat, pvalue = levene(test_df[col_name_1],
                              cont_df[col_name_2])
    print(pvalue)

var_homo("Purchase", "Purchase")

# H1: Varyanslar homojen dağılmamıştır.          p-value > 0.05 ise RED


# Adım 2: Normallik Varsayımı ve Varyans Homojenliği sonuçlarına göre uygun testi seçiniz

# H1: Dağılım normal değildir.  p-value > 0.05 ise RED VE H1: Varyanslar homojen dağılmamıştır.
# Parametric test uygulanmalıdır.

non_stats, pvalue = ttest_ind(cont_df["Purchase"], test_df["Purchase"], equal_var=True)
print(pvalue)

# Adım 3: Test sonucunda elde edilen p_value değerini göz önünde bulundurarak kontrol ve test grubu satın alma
# ortalamaları arasında istatistiki olarak anlamlı bir fark olup olmadığını yorumlayınız.

# Test sonucu p-value > 0.05.
# H0 geçerlidir yani iki durum arasında anlam farkı yoktur (istatistiksel olarak).


##############################################################
# GÖREV 4 : Sonuçların Analizi
##############################################################

# Adım 1: Hangi testi kullandınız, sebeplerini belirtiniz.


# Normallik varsayımını test etmek için Shapiro testini,
# Varyans Homojenliği varsayımını test etmek için Levene testini kullandım.
# parametric test yardımıyla AB test işlemini tamamladım.

# Adım 2: Elde ettiğiniz test sonuçlarına göre müşteriye tavsiyede bulununuz.

# Grupların ortalamaları arasında anlamlı(istatiksel) bir fark yoktur.