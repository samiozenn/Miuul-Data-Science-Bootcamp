############# Feature Engineering on Diabetes Dataset #############

# Özellikleri belirtildiğinde kişilerin diyabet hastası olup olmadıklarını tahmin
# edebilecek bir makine öğrenmesi modeli geliştirilmesi istenmektedir. Modeli
# geliştirmeden önce gerekli olan veri analizi ve özellik mühendisliği adımlarını
# gerçekleştirmeniz beklenmektedir.

# Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin parçasıdır. ABD'deki
# Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan Pima Indian kadınları üzerinde
# yapılan diyabet araştırması için kullanılan verilerdir.
# Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet test sonucunun pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir.

## Task 1 : Keşifçi Veri Analizi

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# Step 1:  Genel resmi inceleyiniz

df = pd.read_csv("6. Hafta/Datasets/diabetes.csv")

df.head()
df.shape
df.describe().T
df.isnull().sum() * df.shape[0] / 100
df.info()
df.dtypes

# Adım 2: Numerik ve kategorik değişkenleri yakalayınız.

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız.

df[num_cols].head()
df[num_cols].describe().T
# df[num_cols].value_counts()

df[num_cols].isnull().sum() / df.shape[0] * 100
df[num_cols].info()
df[num_cols].dtypes

df[cat_cols].head()
df[cat_cols].describe().T
df[cat_cols].info()
df[cat_cols].dtypes

# Adım 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre
# numerik değişkenlerin ortalaması)

# Kategorik değişkenlere göre hedef değişkenin ortalaması
df.groupby("Outcome")["Outcome"].mean()

# hedef değişkene göre numerik değişkenlerin ortalaması
df.groupby("Pregnancies")["Outcome"].mean().head()
df.groupby("Glucose")["Outcome"].mean().head()
df.groupby("BloodPressure")["Outcome"].mean().head()
df.groupby("SkinThickness")["Outcome"].mean().head()
df.groupby("Insulin")["Outcome"].mean().head()
df.groupby("BMI")["Outcome"].mean().head()
df.groupby("DiabetesPedigreeFunction")["Outcome"].mean().head()
df.groupby("Age")["Outcome"].mean().head()

# Adım 5: Aykırı gözlem analizi yapınız.

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, col))

# Adım 6: Eksik gözlem analizi yapınız.


# eksik gozlem var mı yok mu sorgusu
df.isnull().values.any()

# degiskenlerdeki eksik deger sayisi
df.isnull().sum()

# degiskenlerdeki tam deger sayisi
df.notnull().sum()

# veri setindeki toplam eksik deger sayisi
df.isnull().sum().sum()

# en az bir tane eksik degere sahip olan gözlem birimleri
df[df.isnull().any(axis=1)]

# tam olan gözlem birimleri
df[df.notnull().all(axis=1)]

# Adım 7: Korelasyon analizi yapınız
# df.corr().drop("Outcome", axis=0)
df.corr().sort_values("Outcome", ascending=False).drop("Outcome", axis=0)
df.corr().drop("Outcome", axis=0)



## Görev 2 : Feature Engineering

# Adım 1: Eksik ve aykırı değerler için gerekli işlemleri yapınız. Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin vb.
# değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir. Örneğin; bir kişinin glikoz veya insulin değeri 0
# olamayacaktır. Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp sonrasında eksik
# değerlere işlemleri uygulayabilirsiniz.

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df, True)
nan_cols = [col for col in df.columns if  df[col].nunique() > 20]
df[nan_cols].nunique()
df[nan_cols] = df[nan_cols].replace(0, np.nan)
missing_values_table(df, True)
df[nan_cols].isnull().sum()

df.head()
df.describe().T

for col in nan_cols:
    df.loc[(df[col].isnull()) & (df["Outcome"] == 0), col] = df.groupby("Outcome")[col].mean()[0]
    df.loc[(df[col].isnull()) & (df["Outcome"] == 1), col] = df.groupby("Outcome")[col].mean()[1]

df.describe().T

# df[nan_cols].isnull().sum()


missing_values_table(df, True)


for col in num_cols:
    print(col, check_outlier(df, col))

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    replace_with_thresholds(df, col)


for col in num_cols:
    print(col, check_outlier(df, col))


df.describe().T
df.info()


# Adım 2: Yeni değişkenler oluşturunuz.

# Yaşa göre yeni bir değişken oluşturdum.


df.loc[(df['Age'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['Age'] >= 56), 'NEW_AGE_CAT'] = 'senior'

# BMI nedir ? Vücut kitle endeksidir.
# 18.5'un altındaysanız aşırı zayıfsınız,
# 18.5-24.9 arasıysanız normalsiniz,
# 25-29.9 arasındaysanız kilolusunuz,
# 30'dan fazlaysanız obezsiniz demektir.

# BMI'a göre yeni bir değişken oluşturdum.
df.loc[(df['BMI'] < 18.5), 'NEW_BMI_CAT'] = 'underweight'
df.loc[(df['BMI'] >= 18.5) & (df['BMI'] < 25), 'NEW_BMI_CAT'] = 'healthyweight'
df.loc[(df['BMI'] >= 25) & (df['BMI'] < 30), 'NEW_BMI_CAT'] = 'overweight'
df.loc[(df['BMI'] >= 30), 'NEW_BMI_CAT'] = 'obese'

df.head(20)

# New age değişkenini analiz ediyoruz.
df.groupby("NEW_AGE_CAT")["Outcome"].mean()
df["NEW_AGE_CAT"].value_counts()

# New BMI değişkenini analiz ediyoruz.
df.groupby("NEW_BMI_CAT")["Outcome"].mean()
df["NEW_BMI_CAT"].value_counts()
df.head()
df.info()
df.nunique()

df.loc[(df['BMI'] < 18.5) & (df['Age'] < 56), 'NEW_AGE_BMI'] = 'matureunderweight'
df.loc[(df['BMI'] >= 18.5) & (df['Age'] < 56), 'NEW_AGE_BMI'] = 'maturehealthyweight'
df.loc[(df['BMI'] >= 25) & (df['Age'] < 56), 'NEW_AGE_BMI'] = 'matureoverweight'
df.loc[(df['BMI'] >= 30) & (df['Age'] < 56), 'NEW_AGE_BMI'] = 'matureobese'

df.loc[(df['BMI'] < 18.5) & (df['Age'] >= 56), 'NEW_AGE_BMI'] = 'seniorunderweight'
df.loc[(df['BMI'] >= 18.5) & (df['Age'] >= 56), 'NEW_AGE_BMI'] = 'seniorhealthyweight'
df.loc[(df['BMI'] >= 25) & (df['Age'] >= 56), 'NEW_AGE_BMI'] = 'senioroverweight'
df.loc[(df['BMI'] >= 30) & (df['Age'] >= 56), 'NEW_AGE_BMI'] = 'seniorobese'

df.groupby("NEW_AGE_BMI")["Outcome"].mean()
df["NEW_AGE_BMI"].value_counts()

# Adım 3: Encoding işlemlerini gerçekleştiriniz.
df.nunique()
"""
le = LabelEncoder()
le.fit_transform(df["Outcome"])[0:5]
le.inverse_transform([0, 1])


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

df = load()

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)
"""

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique()]

one_hot_encoder(df, ohe_cols).head()

# df.head()
# df.nunique()
df.info()

# Adım 4: Numerik değişkenler için standartlaştırma yapınız.

rs = RobustScaler()

for col in num_cols:
    df[str(col) + "_robuts_scaler"] = rs.fit_transform(df[[col]])


df.describe().T
df.info()
df.head()
"""
df.groupby("NEW_AGE_BMI")["Outcome"].mean()
df["NEW_AGE_BMI"].head()
df["NEW_AGE_BMI"].value_counts()
"""
# Adım 5: Model oluşturunuz.

# df = pd.read_csv("6. Hafta/Datasets/diabetes.csv")


y = df["Outcome"]
X = df.drop(["Outcome","NEW_AGE_CAT", "NEW_BMI_CAT","NEW_AGE_BMI"], axis=1)
# X = df.drop(["Outcome"], axis=1)
X.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)
