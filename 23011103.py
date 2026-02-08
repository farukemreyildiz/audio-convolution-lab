import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import os

#Kod başladığı zaman sizden bir seçim istiyor(1 veya 2) o seçime göre ilerliyoruz.

# ---------------------------------------------------------
# SES İŞLEMLERİ İÇİN YARDIMCI FONKSİYONLAR
# ---------------------------------------------------------
def ses_kaydi_al(ornekleme, sure):
    """Kullanıcının belirlediği süre boyunca ses kaydı alır."""
    print(f"{sure} saniyelik ses kaydı başlatılıyor...")
    kayit = sd.rec(int(sure * ornekleme), samplerate=ornekleme, channels=1, dtype="float64")
    sd.wait()
    print("Ses kaydı tamamlandı.")
    return kayit[:, 0]

def Sistem_cevabı(M, A, delay):
    """Cevabı oluşturur"""
    length = delay * M + 1
    h = np.zeros(length)
    h[0] = 1
    for k in range(1, M + 1):
        h[delay * k] = A * k
    return h


def ses_grafik_ciz(s, ornekleme, title="Ses Grafiği"):
    """5 ve 10 saniyelik girdi seslerin grafiğini çizer """
    t = np.arange(len(s)) / ornekleme

    fig, axs = plt.subplots(2, 1, figsize=(12, 6))

    # Dalga formu grafiği
    axs[0].plot(t, s, color="blue")
    axs[0].set_title("Dalga Formu - " + title)
    axs[0].set_xlabel("Zaman (s)")
    axs[0].set_ylabel("Genlik")
    axs[0].grid(True)

    # Spektrogram grafiği
    axs[1].specgram(s, Fs=ornekleme, cmap="viridis")
    axs[1].set_title("Spektrogram - " + title)
    axs[1].set_xlabel("Zaman (s)")
    axs[1].set_ylabel("Frekans (Hz)")

    plt.tight_layout()
    plt.show()


def konvolusion_sonuc_grafik(M, ornekleme, sonuc_ses1, sonuc_ses2, numpy_sonuc1, numpy_sonuc2):
    """Konvolüsyon sonuçlarının grafiğini çizer"""
    print("Konvolüsyon sonuçlarının grafiklerini çizdiriyor...")
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"M = {M} için Konvolüsyon Sonuçları", fontsize=14)

    # Kendi konvolüsyon (5 saniye)
    t1 = np.arange(len(sonuc_ses1)) / ornekleme
    axs[0, 0].plot(t1, sonuc_ses1, color='blue')
    axs[0, 0].set_title("Kendi Konvolüsyon (5 sn)")
    axs[0, 0].set_xlabel("Zaman (s)")
    axs[0, 0].set_ylabel("Genlik")
    axs[0, 0].grid(True)

    # Kendi konvolüsyon (10 saniye)
    t2 = np.arange(len(sonuc_ses2)) / ornekleme
    axs[0, 1].plot(t2, sonuc_ses2, color='blue')
    axs[0, 1].set_title("Kendi Konvolüsyon (10 sn)")
    axs[0, 1].set_xlabel("Zaman (s)")
    axs[0, 1].set_ylabel("Genlik")
    axs[0, 1].grid(True)

    # NumPy konvolüsyon (5 saniye)
    t3 = np.arange(len(numpy_sonuc1)) / ornekleme
    axs[1, 0].plot(t3, numpy_sonuc1, color='magenta')
    axs[1, 0].set_title("NumPy Konvolüsyon (5 sn)")
    axs[1, 0].set_xlabel("Zaman (s)")
    axs[1, 0].set_ylabel("Genlik")
    axs[1, 0].grid(True)

    # NumPy konvolüsyon (10 saniye)
    t4 = np.arange(len(numpy_sonuc2)) / ornekleme
    axs[1, 1].plot(t4, numpy_sonuc2, color='magenta')
    axs[1, 1].set_title("NumPy Konvolüsyon (10 sn)")
    axs[1, 1].set_xlabel("Zaman (s)")
    axs[1, 1].set_ylabel("Genlik")
    axs[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def ses_islemleri():
    """Ses kaydını alır, işler ve çıktısını verir"""
    ornekleme = 4000

    print("Önce 5 saniyelik ses kaydı alınıyor...")
    ses_kaydi1 = ses_kaydi_al(ornekleme, 5)
    print("Sonrasında 10 saniyelik ses kaydı alınıyor...")
    ses_kaydi2 = ses_kaydi_al(ornekleme, 10)

    ses1 = np.array(ses_kaydi1)
    ses2 = np.array(ses_kaydi2)
    indeks_ses1 = list(range(len(ses1)))
    indeks_ses2 = list(range(len(ses2)))

    # Orijinal ses kayıtlarının grafiklerini göster
    ses_grafik_ciz(ses1, ornekleme, title="5 Saniyelik Ses Kaydı")
    ses_grafik_ciz(ses2, ornekleme, title="10 Saniyelik Ses Kaydı")

    for M in range(3, 6):
        print(f"\nM = {M} için konvolüsyon işlemi başlatıldı.")
        darbe = Sistem_cevabı(M, 0.5, 400)
        indeks_darbe = list(range(len(darbe)))

        # Özel konvolüsyon (çift döngü) ile hesaplama
        sonuc_ses1, _ = ozel_konvHesapla(ses1, indeks_ses1, darbe, indeks_darbe)
        sonuc_ses2, _ = ozel_konvHesapla(ses2, indeks_ses2, darbe, indeks_darbe)
        # Hazır NumPy konvolüsyon hesapları
        numpy_sonuc1 = np.convolve(ses1, darbe, mode="full")
        numpy_sonuc2 = np.convolve(ses2, darbe, mode="full")
        #Süreye etkisini gösterir
        print("Birinci ses kaydının uzunluğu:", len(ses1))
        print("Birinci ses, Özel Konvolüsyon uzunluğu:", len(sonuc_ses1), f"({len(sonuc_ses1) / ornekleme:.2f} s)")
        print("İkinci ses kaydının uzunluğu:", len(ses2))
        print("İkinci ses, Özel Konvolüsyon uzunluğu:", len(sonuc_ses2), f"({len(sonuc_ses2) / ornekleme:.2f} s)")

        # Konvolüsyon sonuçlarını grafikle göster
        konvolusion_sonuc_grafik(M, ornekleme, sonuc_ses1, sonuc_ses2, numpy_sonuc1, numpy_sonuc2)

        # Orijinal sesleri sırayla oynat
        print("\nÖncelikle orijinal 5 saniyelik ses çalınıyor...")
        sd.play(ses1, ornekleme)
        sd.wait()
        print("Öncelikle orijinal 10 saniyelik ses çalınıyor...")
        sd.play(ses2, ornekleme)
        sd.wait()

        tum_sesler = [sonuc_ses1, sonuc_ses2, numpy_sonuc1, numpy_sonuc2]
        for idx, ses_veri in enumerate(tum_sesler, start=1):
            if idx <= 2:
                conv_tip = "Kendi Konvolüsyon (özel hesaplama)"
            else:
                conv_tip = "Hazır Konvolüsyon (NumPy)"
            print(f"M = {M} için {idx}. {conv_tip} ses kaydı çalınıyor...")
            sd.play(ses_veri, ornekleme)
            sd.wait()
            print("Oynatma tamamlandı.")



# ---------------------------------------------------------
# KONVOLÜSYON VE GÖRSEL KARŞILAŞTIRMA İÇİN FONKSİYONLAR
# ---------------------------------------------------------
def ozel_konvHesapla(sinyal1, indeks1, sinyal2, indeks2):
    """
    Verilen iki sinyalin konvolüsyonunu hesaplar;
    sinyal1 ve sinyal2 için başlangıç indekslerini kullanarak çıktının indekslemesini belirler.
    """
    len1 = len(sinyal1)
    len2 = len(sinyal2)
    toplam_uzunluk = len1 + len2 - 1
    konv_sonuc = [0] * toplam_uzunluk

    for a in range(len1):
        for b in range(len2):
            konv_sonuc[a + b] += sinyal1[a] * sinyal2[b]
    baslangic = indeks1[0] + indeks2[0]
    sonuc_indeksleri = [baslangic + k for k in range(toplam_uzunluk)]
    return konv_sonuc, sonuc_indeksleri


def sinirli_dizi_girisi():
    """Kullanıcıdan maksimum 5 eleman alabileceği şekilde iki sinyal (dizi) girdisi sağlar."""
    while True:
        try:
            adet_A = int(input("Birinci dizinin eleman sayısını girin (max 5): "))
            if adet_A > 5:
                print("Birinci dizi 5'ten fazla eleman içeremez, lütfen tekrar deneyin.")
                continue
            sifirA = int(input("Birinci dizide n=0 olan elemanın pozisyonunu girin (1-index): ")) - 1
            sinyal_A = list(map(float, input(f"{adet_A} elemanlı diziyi (boşlukla ayrılarak) girin: ").split()))
            if len(sinyal_A) != adet_A:
                print("Girilen eleman sayısı belirtilen adete uymuyor.")
                continue
            indeks_A = [i - sifirA for i in range(adet_A)]

            adet_B = int(input("İkinci dizinin eleman sayısını girin (max 5): "))
            if adet_B > 5:
                print("İkinci dizi 5'ten fazla eleman içeremez, lütfen tekrar deneyin.")
                continue
            sifirB = int(input("İkinci dizide n=0 olan elemanın pozisyonunu girin (1-index): ")) - 1
            sinyal_B = list(map(float, input(f"{adet_B} elemanlı diziyi (boşlukla ayrılarak) girin: ").split()))
            if len(sinyal_B) != adet_B:
                print("Girilen eleman sayısı belirtilen adete uymuyor.")
                continue
            indeks_B = [i - sifirB for i in range(adet_B)]

            return sinyal_A, indeks_A, sinyal_B, indeks_B
        except ValueError:
            print("Geçersiz giriş tespit edildi, lütfen yalnızca sayısal değerler girin.")


def vektor_karsilastir(sinyal_A, sinyal_B, conv_ozel, conv_numpy):
    """Konsola; orijinal sinyaller ve konvolüsyon sonuçlarını yazdırır."""
    print("\n------------------------------")
    print("Sinyal A:", sinyal_A)
    print("Sinyal B:", sinyal_B)
    print("Özel Konvolüsyon Sonucu:", conv_ozel)
    print("Numpy Konvolüsyon Sonucu:", conv_numpy)


def grafik_goster(conv_ozel, indeks_ozel, conv_numpy, indeks_numpy, sinyal_A, indeks_A, sinyal_B, indeks_B):
    """Dört ayrı grafik penceresinde; giriş sinyalleri ve hesaplanan konvolüsyon sonuçlarını sunar."""
    fig, eksenler = plt.subplots(4, 1, figsize=(10, 20))

    veri_setleri = [
        (sinyal_A, indeks_A, "Girdi Sinyali A[n]", "red", "ro"),
        (sinyal_B, indeks_B, "Girdi Sinyali B[n]", "green", "go"),
        (conv_ozel, indeks_ozel, "Özel Konvolüsyon Sonucu", "blue", "bo"),
        (conv_numpy, indeks_numpy, "Numpy Konvolüsyon Sonucu", "magenta", "mo")
    ]

    for eks, (veri, idx, baslik, renk, isaret) in zip(eksnen := eksenler, veri_setleri):
        eks.stem(idx, veri, linefmt=renk, markerfmt=isaret, basefmt=" ")
        eks.set(title=baslik, xlabel="İndeks", ylabel="Genlik")
        eks.legend([baslik])
        eks.grid(True)
        ozel_tick = np.unique(np.array(veri)[np.array(veri) != 0])
        if ozel_tick.size > 0:
            eks.set_yticks(ozel_tick)
    plt.subplots_adjust(hspace=0.4)
    plt.show()



# ---------------------------------------------------------
# ANA PROGRAM
# ---------------------------------------------------------

#Kod başladığı zaman sizden bir seçim istiyor(1 veya 2) o seçime göre ilerliyoruz.

def main():
    print("Mod Seçimi:")
    print("1 - Dizi Girdisi ile Konvolüsyon")
    print("2 - Ses Kaydı ve İşlem Modu")
    try:
        mod_sec = int(input("Lütfen 1 veya 2'yi seçin: "))
    except ValueError:
        print("Uygun bir sayı giriniz!")
        return

    if mod_sec == 1:
        sinyal_A, indeks_A, sinyal_B, indeks_B = sinirli_dizi_girisi()
        conv_ozel, indeks_ozel = ozel_konvHesapla(sinyal_A, indeks_A, sinyal_B, indeks_B)

        np_sinyal_A = np.array(sinyal_A)
        np_sinyal_B = np.array(sinyal_B)
        conv_numpy = np.convolve(np_sinyal_A, np_sinyal_B, mode="full")
        indeks_numpy = [indeks_A[0] + indeks_B[0] + i for i in range(len(conv_numpy))]

        vektor_karsilastir(sinyal_A, sinyal_B, conv_ozel, conv_numpy)
        grafik_goster(conv_ozel, indeks_ozel, conv_numpy, indeks_numpy, sinyal_A, indeks_A, sinyal_B, indeks_B)
    elif mod_sec == 2:
        ses_islemleri()
        print("Tüm ses işlemleri bitti.")
    else:
        print("Yanlış seçim yaptınız. Lütfen 1 veya 2 giriniz.")


if __name__ == "__main__":
    main()
