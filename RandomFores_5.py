# en düşük feature score a sahip 3 özellik çıkartıldı
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import category_encoders as ce
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

veriseti = pd.read_csv("soru_1.csv") #veri seti projeye dahil edildi

X = veriseti.drop(['sirket','begenmek','yil','yapimci_sayisi'], axis=1)
y = veriseti['sirket']

# eğitim ve test veri setleri oluşturuldu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)

encoder = ce.OrdinalEncoder(cols=['ulke','izleyici','money','imdb'])   # sözel bilgiler sayısal bilgilere dönüştürüldü
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

model = RandomForestClassifier(n_estimators=100,random_state=0) # sınıflandırıcı model oluşturuldu(n_estimators, ağaç sayısını tutuyor.) 

model.fit(X_train, y_train)  # model eğitim testi ile eğitilir

y_pred = model.predict(X_test) # model test veri ile test edilir.

print('100 Karar ağaçlı model doğruluk skoru :  {0:0.4f}'. format(accuracy_score(y_test, y_pred))) # başarı sonucu ekrana yazdırılır

# veri setindeki özelliklerin önem puanları hesaplandı
feature_scores = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)

print(feature_scores)

# Seaborn grafiği oluşturuldu
sns.barplot(x=feature_scores, y=feature_scores.index)

# Etiket isimleri eklendi
plt.xlabel('Feaute Importance Score')
plt.ylabel('Features')

# Grafiğe başlık eklendi
plt.title("Visualizing Important Features")
plt.show()

# Karmaşıklık matrisi oluşturuldu
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix\n\n', cm)

#classfication report
print(classification_report(y_test, y_pred))