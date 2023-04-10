##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################
import pandas as pd

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir.
# Şirketin orta uzun vadeli plan yapabilmesi için var olan müşterilerin gelecekte şirkete sağlayacakları potansiyel değerin tahmin edilmesi gerekmektedir.


###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi


###############################################################
# GÖREVLER
###############################################################
# GÖREV 1: Veriyi Hazırlama
           # 1. flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.
           # 2. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
           # Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.
           # 3. "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" değişkenlerinin
           # aykırı değerleri varsa baskılayanız.
           # 4. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Herbir müşterinin toplam
           # alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.
           # 5. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

# GÖREV 2: CLTV Veri Yapısının Oluşturulması
           # 1.Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.
           # 2.customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.
           # Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure değerleri ise haftalık cinsten ifade edilecek.


# GÖREV 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması, CLTV'nin hesaplanması
           # 1. BG/NBD modelini fit ediniz.
                # a. 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.
                # b. 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.
           # 2. Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz.
           # 3. 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.
                # b. Cltv değeri en yüksek 20 kişiyi gözlemleyiniz.

# GÖREV 4: CLTV'ye Göre Segmentlerin Oluşturulması
           # 1. 6 aylık tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz. cltv_segment ismi ile dataframe'e ekleyiniz.
           # 2. 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz

# BONUS: Tüm süreci fonksiyonlaştırınız.


###############################################################
# GÖREV 1: Veriyi Hazırlama
###############################################################
# !pip install lifetimes
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler

# 1. OmniChannel.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.
df_ = pd.read_csv("3. Hafta CRM Analizi/Ödevler/FLOCLTVPrediction-230305-185416/FLOCLTVPrediction/flo_data_20k.csv")
df = df_.copy()
df.head()
df.describe().T
df.isnull().sum()
# 2. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = round(quartile3 + 1.5 * interquantile_range)
    low_limit = round(quartile1 - 1.5 * interquantile_range)
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# 3. "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" değişkenlerinin
#aykırı değerleri varsa baskılayanız.

aykiri_degerler = ["order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online"]
for i in aykiri_degerler:
    replace_with_thresholds(df, i)

"""       
replace_with_thresholds(df, "order_num_total_ever_online")
replace_with_thresholds(df, "order_num_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_online")
"""
df.describe([0.1,0.5,0.99]).T


# 4. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir.
# Herbir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.
df["num_total_ever_omnichannel_"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total_omnichannel"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]


# 5. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
df.head()
df.info()

dates = df.columns[df.columns.str.contains("date")]
# [df.columns.str.contains("date")] çıktısı = [False, False, False,  True,  True,  True,  True, False, False,
#                                              False, False, False, False, False]
# df.columns[df.columns.str.contains("date")] çıktısı = ['first_order_date', 'last_order_date',
#
df[dates] = df[dates].apply(pd.to_datetime)
"""
df["last_order_date"] = df["last_order_date"].apply(pd.to_datetime)
df["first_order_date"] = df["first_order_date"].apply(pd.to_datetime)
df["last_order_date_online"] = df["last_order_date_online"].apply(pd.to_datetime)
df["last_order_date_offline"] = df["last_order_date_offline"].apply(pd.to_datetime)
"""
df.info()
###############################################################
# GÖREV 2: CLTV Veri Yapısının Oluşturulması
###############################################################
# recency: Son satın alma üzerinden geçen zaman. Haftalık. (kullanıcı özelinde)
# T: Müşterinin yaşı. Haftalık. (analiz tarihinden ne kadar süre önce ilk satın alma yapılmış)
# frequency: tekrar eden toplam satın alma sayısı (frequency>1)
# monetary: satın alma başına ortalama kazanç

# 1.Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.
df["last_order_date"].max()
analyz_date = dt.datetime(2021, 6, 1)
df.describe().T
# 2.customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.
cltv = pd.DataFrame(columns=["customer_id", "recency_cltv_weekly", "T_weekly", "frequency" , "monetary_cltv_avg"])
cltv["customer_id"] = df["master_id"]
cltv["recency_cltv_weekly"] = round(((df["last_order_date"] - df["first_order_date"]).astype('timedelta64[D]')) / 7)
cltv["T_weekly"] = round(((analyz_date - df["first_order_date"]).astype('timedelta64[D]')) / 7)
cltv["frequency"] = df["num_total_ever_omnichannel_"]
cltv["monetary_cltv_avg"] = (df["customer_value_total_omnichannel"] / df["num_total_ever_omnichannel_"])


"""
cltv = df.groupby("master_id").agg({"last_order_date" : ["max"],
                                    "first_order_date":["min",
                                    lambda x : (analyz_date - x.min()).days],
                                   "num_total_ever_omnichannel_" : lambda x: x.sum(),
                                   "customer_value_total_omnichannel" : lambda x: x.sum()})

cltv.columns = cltv.columns.droplevel(0)
cltv.reset_index(inplace=True)
"""

# cltv["recency_cltv_weekly"] = round(((cltv["max"] - cltv["min"]).astype('timedelta64[D]')) / 7)
# ((cltv["max"] - cltv["min"])/7) böyle yazılırsa hata alınır.


cltv.info()
cltv.head()
cltv.describe().T

###############################################################
# GÖREV 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması, 6 aylık CLTV'nin hesaplanması
###############################################################

# 1. BG/NBD modelini kurunuz.

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv["frequency"],
        cltv["recency_cltv_weekly"],
        cltv["T_weekly"])
# 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.

cltv["exp_sales_3_month"] = bgf.predict(3*4,
                                        cltv["frequency"],
                                        cltv["recency_cltv_weekly"],
                                        cltv["T_weekly"]
                                        )
# 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.


cltv["exp_sales_6_month"] = bgf.predict(6*4,
                                        cltv["frequency"],
                                        cltv["recency_cltv_weekly"],
                                        cltv["T_weekly"]
                                        )
"""
bgf.predict(60*4,
                                        cltv["frequency"],
                                        cltv["recency_cltv_weekly"],
                                        cltv["T_weekly"]
                                        )
                                        # tahmin modeli doğrusal çalışıyor 
"""
# 3. ve 6.aydaki en çok satın alım gerçekleştirecek 10 kişiyi inceleyeniz.
bgf.predict(3*4,
                                        cltv["frequency"],
                                        cltv["recency_cltv_weekly"],
                                        cltv["T_weekly"]
                                        ).sort_values(ascending=False).head(10)

bgf.predict(6*4,
                                        cltv["frequency"],
                                        cltv["recency_cltv_weekly"],
                                        cltv["T_weekly"]
                                        ).sort_values(ascending=False).head(10)
#


# 2.  Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz.

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv['frequency'], cltv['monetary_cltv_avg'])

cltv["exp_average_value"] = ggf.conditional_expected_average_profit(cltv['frequency'],
                                                                    cltv['monetary_cltv_avg'])
# 3. 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.

cltv["cltv"] = ggf.customer_lifetime_value(bgf,
                                   cltv['frequency'],
                                   cltv['recency_cltv_weekly'],
                                   cltv['T_weekly'],
                                   cltv['monetary_cltv_avg'],
                                   time=6,  # 6 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)
"""
ggf.customer_lifetime_value(bgf,
                                   cltv['frequency'],
                                   cltv['recency_cltv_weekly'],
                                   cltv['T_weekly'],
                                   cltv['monetary_cltv_avg'],
                                   time=60,  # 60 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)
                                   # buradaki model doğrusal değil + sonsuzda eğimi 0 a yaklaşabilir
"""



# CLTV değeri en yüksek 20 kişiyi gözlemleyiniz.

cltv.sort_values(by= "cltv",ascending= False).head(20)


###############################################################
# GÖREV 4: CLTV'ye Göre Segmentlerin Oluşturulması
###############################################################

# 1. 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.
# cltv_segment ismi ile atayınız.

cltv["segment"] = pd.qcut(cltv["cltv"], 4, labels=["D", "C", "B", "A"])


# 2. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.


# cltv.groupby("segment")["recency_cltv_weekly","frequency", "monetary_cltv_avg"].agg(["mean","sum"])

cltv.groupby("segment").agg(
    { "mean", "sum"})




