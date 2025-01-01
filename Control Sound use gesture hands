import cv2
import pyautogui
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import mediapipe as mp
import numpy as np
import math

# Konfigurasi
MIN_DET_EKSI = 0.5
MIN_TRACK_EKSI = 0.5

# Inisialisasi Mediapipe
mp_tangan = mp.solutions.hands
tangan = mp_tangan.Hands(min_detection_confidence=MIN_DET_EKSI, min_tracking_confidence=MIN_TRACK_EKSI)
mp_drawing = mp.solutions.drawing_utils

# Inisialisasi OpenCV
cap = cv2.VideoCapture(0)

# Dapatkan objek volume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_,
    CLSCTX_ALL,
    None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Dapatkan rentang volume saat ini
volume_range = volume.GetVolumeRange()
volume_min = volume_range[0]
volume_max = volume_range[1]

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Flip horizontal
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hasil = tangan.process(rgb_frame)

    if hasil.multi_hand_landmarks:
        for tangan_landmark in hasil.multi_hand_landmarks:
            # Mengidentifikasi jari
            ibu_jari = tangan_landmark.landmark[mp_tangan.HandLandmark.THUMB_TIP]
            telunjuk = tangan_landmark.landmark[mp_tangan.HandLandmark.INDEX_FINGER_TIP]

            # Konversi ke koordinat piksel
            h, w, c = frame.shape
            ibu_jari_x, ibu_jari_y = int(ibu_jari.x * w), int(ibu_jari.y * h)
            telunjuk_x, telunjuk_y = int(telunjuk.x * w), int(telunjuk.y * h)

            # Menghitung jarak Euclidean antara ibu jari dan telunjuk
            jarak = math.hypot(telunjuk_x - ibu_jari_x, telunjuk_y - ibu_jari_y)

            # Skala jarak ke rentang volume
            # Atur jarak_maks ke nilai yang sesuai dengan rentang jari yang nyaman
            jarak_min = 50  # jarak minimal yang mungkin antara ibu jari dan telunjuk
            jarak_maks = 300  # jarak maksimal yang mungkin antara ibu jari dan telunjuk
            volume_baru = np.interp(jarak, [jarak_min, jarak_maks], [volume_min, volume_max])

            # Mengatur volume
            volume.SetMasterVolumeLevel(volume_baru, None)

            # Menggambar garis dan titik
            cv2.circle(frame, (ibu_jari_x, ibu_jari_y), 10, (0, 255, 0), cv2.FILLED)
            cv2.circle(frame, (telunjuk_x, telunjuk_y), 10, (0, 255, 0), cv2.FILLED)
            cv2.line(frame, (ibu_jari_x, ibu_jari_y), (telunjuk_x, telunjuk_y), (0, 255, 0), 3)

            mp_drawing.draw_landmarks(frame, tangan_landmark, mp_tangan.HAND_CONNECTIONS)

    cv2.imshow("Gesture Control", frame)

    # Keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup kamera dan jendela
cap.release()
cv2.destroyAllWindows()
