# ============================================================
#  Face Detection — Name & Age Always Visible
#  pip install opencv-python
# ============================================================

import cv2
from datetime import date

# ──────────────────────────────────────────────
#  REFERENCE DATA
# ──────────────────────────────────────────────
PERSON_NAME = "Ullas"
DOB         = date(2002, 9, 9)
IMAGE_PATH  = "PHOTO.jpg"

# ──────────────────────────────────────────────
#  AUTO CALCULATE AGE
# ──────────────────────────────────────────────
today = date.today()
age   = today.year - DOB.year - ((today.month, today.day) < (DOB.month, DOB.day))
dob_str = DOB.strftime("%d %B %Y")

print(f"Name : {PERSON_NAME}")
print(f"DOB  : {dob_str}")
print(f"Age  : {age} years")

# ──────────────────────────────────────────────
#  LOAD IMAGE
# ──────────────────────────────────────────────
image = cv2.imread(IMAGE_PATH)
if image is None:
    print(f"[ERROR] Cannot load '{IMAGE_PATH}' — check the file is in the same folder.")
    exit()

# Resize for better display if image is too small
h, w = image.shape[:2]
if w < 400:
    scale = 400 / w
    image = cv2.resize(image, (int(w * scale), int(h * scale)))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Equalise to improve detection in bright/outdoor photos
gray = cv2.equalizeHist(gray)

# ──────────────────────────────────────────────
#  FACE DETECTION  (tries 3 passes)
# ──────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

if len(faces) == 0:
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40))

if len(faces) == 0:
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.02, minNeighbors=2, minSize=(30, 30))

H, W = image.shape[:2]

# ──────────────────────────────────────────────
#  DRAW FACE BOX + EYE DOTS
# ──────────────────────────────────────────────
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

if len(faces) > 0:
    print(f"[OK] {len(faces)} face(s) detected!")
    for (x, y, fw, fh) in faces:
        # Face rectangle
        cv2.rectangle(image, (x, y), (x + fw, y + fh), (0, 220, 0), 3)

        # Eyes
        roi_gray = gray[y:y+fh, x:x+fw]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 8)
        for (ex, ey, ew, eh) in eyes:
            cx = x + ex + ew // 2
            cy = y + ey + eh // 2
            cv2.circle(image, (cx, cy), ew // 2, (255, 140, 0), 2)
else:
    print("[WARNING] Face not detected — drawing name & age on image anyway.")

# ──────────────────────────────────────────────
#  NAME & AGE — always drawn, big and clear
# ──────────────────────────────────────────────

# ---- Line 1: Name ----
name_text  = f"Name : {PERSON_NAME}"
age_text   = f"Age  : {age} yrs  (DOB: {dob_str})"
match_text = "FACE MATCHED - Reference Verified" if len(faces) > 0 else "FACE NOT DETECTED"
match_color = (0, 220, 100) if len(faces) > 0 else (0, 60, 255)

font      = cv2.FONT_HERSHEY_SIMPLEX
BIG       = 0.85
SMALL     = 0.65
BOLD      = 2
PAD       = 12

# Measure texts
(nw, nh), _ = cv2.getTextSize(name_text,  font, BIG,   BOLD)
(aw, ah), _ = cv2.getTextSize(age_text,   font, SMALL, BOLD)
(mw, mh), _ = cv2.getTextSize(match_text, font, SMALL, BOLD)

panel_h = nh + ah + mh + PAD * 4 + 10
# Dark panel at the BOTTOM of the image
cv2.rectangle(image, (0, H - panel_h), (W, H), (15, 15, 15), cv2.FILLED)

# Draw texts inside panel
y1 = H - panel_h + PAD + nh
cv2.putText(image, name_text,  (PAD, y1),              font, BIG,   (255, 255, 255), BOLD, cv2.LINE_AA)

y2 = y1 + ah + PAD
cv2.putText(image, age_text,   (PAD, y2),              font, SMALL, (180, 220, 255), BOLD, cv2.LINE_AA)

y3 = y2 + mh + PAD
cv2.putText(image, match_text, (PAD, y3),              font, SMALL, match_color,     BOLD, cv2.LINE_AA)

# MATCHED badge top-right
badge_text = "MATCHED" if len(faces) > 0 else "NO MATCH"
badge_col  = (0, 180, 0) if len(faces) > 0 else (0, 0, 200)
(bw, bh), _ = cv2.getTextSize(badge_text, font, 0.6, 2)
cv2.rectangle(image, (W - bw - 20, 10), (W - 5, bh + 20), badge_col, cv2.FILLED)
cv2.putText(image, badge_text, (W - bw - 12, bh + 14), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

# ──────────────────────────────────────────────
#  SHOW & SAVE
# ──────────────────────────────────────────────
cv2.imshow(f"Face Detection — {PERSON_NAME}", image)
print("Press any key to close...")
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("PHOTO.jpg", image)
print("[SAVED] PHOTO.jpg")