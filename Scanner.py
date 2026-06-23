import os
import threading
from io import BytesIO

import customtkinter as ctk
from tkinter import filedialog, messagebox, Canvas
from PIL import Image, ImageTk
import numpy as np
import cv2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
import pytesseract

try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# =====================================
# CONFIG
# =====================================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

DISPLAY_SIZE = 500

ACCENT = "#3B82F6"
CARD_BG = "#1F2433"
SIDEBAR_BG = "#161B26"
SUCCESS_COLOR = "#4ADE80"
ERROR_COLOR = "#F87171"
INFO_COLOR = "#9CA3AF"

LANG_OPTIONS = {
    "Indonesia + English": "ind+eng",
    "English": "eng",
    "Indonesia": "ind",
}

current_image = None
processed_image = None
original_processed = None
adjustment_base = None
pages = []

manual_crop_mode = False
manual_points = []
display_scale = 1.0
manual_mode_default_color = None
ocr_busy = False

# =====================================
# IMAGE PROCESSING HELPERS (pure, no GUI)
# =====================================


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    rect = order_points(np.array(pts, dtype="float32"))
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array(
        [
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1],
        ],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))


def find_document_quad(thresh_img, min_area=5000):
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue

        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)

        for eps_frac in (0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1):
            approx = cv2.approxPolyDP(hull, eps_frac * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2), False

    for c in contours:
        if cv2.contourArea(c) >= min_area:
            box = cv2.boxPoints(cv2.minAreaRect(c))
            return box, True

    return None, False


def binarize_scan(img):
    gray_scan = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return cv2.adaptiveThreshold(
        gray_scan,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,
        10,
    )


def auto_enhance_document(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

    p_low, p_high = np.percentile(gray, (2, 98))
    if p_high - p_low < 1:
        stretched = gray
    else:
        stretched = np.clip(
            (gray.astype(np.float32) - p_low) * (255.0 / (p_high - p_low)), 0, 255
        ).astype(np.uint8)

    denoised = cv2.bilateralFilter(stretched, 9, 75, 75)

    blurred = cv2.GaussianBlur(denoised, (0, 0), 3)
    sharpened = cv2.addWeighted(denoised, 1.5, blurred, -0.5, 0)

    return binarize_scan(sharpened)


def deskew_image(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))

    if coords.shape[0] == 0:
        return img, 0.0

    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle

    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    border_value = 255 if len(img.shape) == 2 else (255, 255, 255)
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )

    return rotated, angle


def draw_fitted_image(pdf, img_bgr, page_w, page_h, margin=36):
    ok, buf = cv2.imencode(".jpg", img_bgr)
    reader = ImageReader(BytesIO(buf.tobytes()))
    iw, ih = reader.getSize()

    avail_w = page_w - 2 * margin
    avail_h = page_h - 2 * margin
    scale = min(avail_w / iw, avail_h / ih)
    draw_w, draw_h = iw * scale, ih * scale
    x = (page_w - draw_w) / 2
    y = (page_h - draw_h) / 2

    pdf.drawImage(reader, x, y, width=draw_w, height=draw_h)


# =====================================
# STATUS / BUSY HELPERS
# =====================================


def set_status(text, color=None):
    status_label.configure(text=text, text_color=color or INFO_COLOR)


def set_busy(is_busy, message=None):
    global ocr_busy
    ocr_busy = is_busy

    if is_busy:
        progress_bar.pack(side="left", padx=(15, 0))
        progress_bar.start()
        if message:
            set_status(message, INFO_COLOR)
    else:
        progress_bar.stop()
        progress_bar.pack_forget()


# =====================================
# DISPLAY
# =====================================


def show_original(img):
    global display_scale

    h, w = img.shape[:2]
    scale = min(DISPLAY_SIZE / w, DISPLAY_SIZE / h, 1.0)
    disp_w, disp_h = max(1, int(w * scale)), max(1, int(h * scale))
    display_scale = scale

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb).resize((disp_w, disp_h))
    photo = ImageTk.PhotoImage(img_pil)

    original_canvas.configure(width=disp_w, height=disp_h)
    original_canvas.delete("all")
    original_canvas.create_image(0, 0, anchor="nw", image=photo, tags="img")
    original_canvas.image = photo

    reset_manual_points()


def show_result(img, update_base=True):
    global processed_image, original_processed, adjustment_base

    processed_image = img.copy()

    if original_processed is None:
        original_processed = img.copy()

    if update_base:
        adjustment_base = img.copy()
        brightness_slider.set(0)
        brightness_value_label.configure(text="0")
        contrast_slider.set(1)
        contrast_value_label.configure(text="1.0")

    if len(img.shape) == 2:
        display = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_pil = Image.fromarray(display)
    img_pil.thumbnail((DISPLAY_SIZE, DISPLAY_SIZE))
    photo = ImageTk.PhotoImage(img_pil)

    result_label.configure(image=photo, text="")
    result_label.image = photo


# =====================================
# FILE LOADING (dialog + drag & drop)
# =====================================


def open_image_from_path(path):
    global current_image, original_processed, adjustment_base

    image = cv2.imread(path)

    if image is None:
        messagebox.showerror("Error", "Gagal membaca gambar")
        return

    current_image = image
    original_processed = None
    adjustment_base = None

    show_original(current_image)
    set_status(f"Gambar berhasil dimuat: {os.path.basename(path)}", SUCCESS_COLOR)


def load_image():
    path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )

    if not path:
        return

    open_image_from_path(path)


def parse_first_dropped_path(raw):
    raw = raw.strip()

    if raw.startswith("{"):
        end = raw.find("}")
        if end != -1:
            return raw[1:end]

    return raw.split(" ")[0]


def handle_drop(event):
    path = parse_first_dropped_path(event.data)

    if not path.lower().endswith((".jpg", ".jpeg", ".png")):
        messagebox.showerror("Error", "Format file tidak didukung (gunakan jpg/jpeg/png)")
        return

    open_image_from_path(path)


# =====================================
# AUTO SCAN
# =====================================


def start_new_scan_result(img):
    global original_processed

    original_processed = None
    show_result(img)


def auto_scan():
    global current_image

    if current_image is None:
        messagebox.showwarning("Peringatan", "Muat gambar terlebih dahulu")
        return

    image = current_image.copy()
    ratio = image.shape[0] / 500.0

    resized = cv2.resize(image, (int(image.shape[1] / ratio), 500))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5
    )
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    document, used_fallback = find_document_quad(thresh)

    if document is None:
        messagebox.showerror(
            "Error",
            "Dokumen tidak ditemukan. Coba 'Mode Crop Manual' untuk memilih sudut secara manual.",
        )
        return

    pts = document.astype("float32") * ratio

    warped = four_point_transform(image, pts)

    start_new_scan_result(warped)
    enhanced = auto_enhance_document(warped)
    show_result(enhanced)

    if used_fallback:
        set_status(
            "Dokumen dicrop & ditingkatkan otomatis (deteksi perkiraan: kecerahan, kontras, ketajaman, hitam-putih). Periksa hasil, atau gunakan 'Mode Crop Manual' jika kurang tepat.",
            SUCCESS_COLOR,
        )
    else:
        set_status(
            "Dokumen berhasil dicrop & ditingkatkan otomatis (kecerahan, kontras, ketajaman, hitam-putih).",
            SUCCESS_COLOR,
        )


# =====================================
# MANUAL CROP
# =====================================


def toggle_manual_crop():
    global manual_crop_mode

    if current_image is None:
        messagebox.showwarning("Peringatan", "Muat gambar terlebih dahulu")
        return

    manual_crop_mode = not manual_crop_mode
    reset_manual_points()

    if manual_crop_mode:
        btn_manual_mode.configure(fg_color="#0EA5E9", text="🖱️ Mode Crop Manual (ON)")
        set_status("Klik 4 sudut dokumen pada gambar Original", INFO_COLOR)
    else:
        btn_manual_mode.configure(fg_color=manual_mode_default_color, text="🖱️ Mode Crop Manual")
        show_original(current_image)
        set_status("Mode crop manual dimatikan", INFO_COLOR)


def on_canvas_click(event):
    if not manual_crop_mode or current_image is None:
        return

    if len(manual_points) >= 4:
        return

    manual_points.append((event.x, event.y))
    draw_manual_overlay()

    if len(manual_points) == 4:
        btn_apply_manual.configure(state="normal")
        set_status("4 titik dipilih. Klik 'Terapkan'.", INFO_COLOR)


def draw_manual_overlay():
    original_canvas.delete("manual")

    for (x, y) in manual_points:
        r = 5
        original_canvas.create_oval(
            x - r, y - r, x + r, y + r,
            outline="#22D3EE", width=2, fill="#0EA5E9", tags="manual",
        )

    if len(manual_points) >= 2:
        flat = [coord for pt in manual_points for coord in pt]
        if len(manual_points) == 4:
            flat += list(manual_points[0])
        original_canvas.create_line(*flat, fill="#22D3EE", width=2, tags="manual")


def reset_manual_points():
    global manual_points
    manual_points = []

    if current_image is not None:
        original_canvas.delete("manual")

    btn_apply_manual.configure(state="disabled")


def apply_manual_crop():
    global manual_crop_mode

    if current_image is None or len(manual_points) != 4:
        return

    pts_original = np.array(
        [(x / display_scale, y / display_scale) for (x, y) in manual_points],
        dtype="float32",
    )

    warped = four_point_transform(current_image, pts_original)

    manual_crop_mode = False
    btn_manual_mode.configure(fg_color=manual_mode_default_color, text="🖱️ Mode Crop Manual")
    show_original(current_image)

    start_new_scan_result(warped)
    set_status(
        "Dokumen berhasil dicrop manual. Pilih filter (Gray/B-W/dll) untuk mempertegas.",
        SUCCESS_COLOR,
    )


# =====================================
# TRANSFORM / RESET
# =====================================


def rotate_image():
    global processed_image

    if processed_image is None:
        return

    processed_image = cv2.rotate(processed_image, cv2.ROTATE_90_CLOCKWISE)
    show_result(processed_image)
    set_status("Gambar dirotasi 90°", SUCCESS_COLOR)


def auto_deskew():
    if processed_image is None:
        messagebox.showwarning("Peringatan", "Belum ada hasil scan untuk diluruskan")
        return

    rotated, angle = deskew_image(processed_image)

    if abs(angle) < 0.1:
        set_status("Gambar sudah lurus (skew ~0°)", INFO_COLOR)
        return

    show_result(rotated)
    set_status(f"Auto Deskew: kemiringan {angle:.2f}° dikoreksi", SUCCESS_COLOR)


def reset_image():
    if original_processed is None:
        return

    show_result(original_processed)
    set_status("Gambar direset", INFO_COLOR)


def apply_brightness_contrast():
    global processed_image

    if adjustment_base is None:
        return

    alpha = float(contrast_slider.get())
    beta = int(float(brightness_slider.get()))

    adjusted = cv2.convertScaleAbs(adjustment_base, alpha=alpha, beta=beta)
    processed_image = adjusted
    show_result(adjusted, update_base=False)


def adjust_brightness(value):
    brightness_value_label.configure(text=str(int(float(value))))
    apply_brightness_contrast()


def adjust_contrast(value):
    contrast_value_label.configure(text=f"{float(value):.1f}")
    apply_brightness_contrast()


# =====================================
# FILTERS
# =====================================


def filter_gray():
    global processed_image

    if processed_image is None:
        return

    if len(processed_image.shape) == 3:
        gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = processed_image.copy()

    show_result(gray)
    set_status("Filter Grayscale diterapkan", SUCCESS_COLOR)


def filter_bw():
    if processed_image is None:
        return

    bw = binarize_scan(processed_image)

    show_result(bw)
    set_status("Filter Black/White (adaptive threshold) diterapkan", SUCCESS_COLOR)


def histogram_equalization():
    global processed_image

    if processed_image is None:
        return

    if len(processed_image.shape) == 3:
        gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = processed_image.copy()

    equalized = cv2.equalizeHist(gray)

    show_result(equalized)
    set_status("Histogram Equalization diterapkan", SUCCESS_COLOR)


def edge_detection():
    global processed_image

    if processed_image is None:
        return

    if len(processed_image.shape) == 3:
        gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = processed_image.copy()

    edges = cv2.Canny(gray, 100, 200)

    show_result(edges)
    set_status("Edge Detection diterapkan", SUCCESS_COLOR)


def filter_sharpen():
    global processed_image

    if processed_image is None:
        return

    blurred = cv2.GaussianBlur(processed_image, (0, 0), 3)
    sharpened = cv2.addWeighted(processed_image, 1.5, blurred, -0.5, 0)

    show_result(sharpened)
    set_status("Filter Sharpen diterapkan", SUCCESS_COLOR)


def filter_denoise():
    global processed_image

    if processed_image is None:
        return

    denoised = cv2.bilateralFilter(processed_image, 9, 75, 75)

    show_result(denoised)
    set_status("Filter Denoise diterapkan", SUCCESS_COLOR)


def filter_sepia():
    global processed_image

    if processed_image is None:
        return

    if len(processed_image.shape) == 2:
        bgr = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = processed_image

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64)

    sepia_matrix = np.array(
        [
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131],
        ]
    )

    sepia_rgb = np.clip(rgb @ sepia_matrix.T, 0, 255).astype(np.uint8)
    sepia_bgr = cv2.cvtColor(sepia_rgb, cv2.COLOR_RGB2BGR)

    show_result(sepia_bgr)
    set_status("Filter Sepia diterapkan", SUCCESS_COLOR)


def filter_invert():
    global processed_image

    if processed_image is None:
        return

    inverted = cv2.bitwise_not(processed_image)

    show_result(inverted)
    set_status("Warna dibalik (Invert)", SUCCESS_COLOR)


def filter_magic_color():
    global processed_image

    if processed_image is None:
        return

    if len(processed_image.shape) == 2:
        bgr = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = processed_image

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=25)
    norm_gray = cv2.divide(gray, background, scale=255)

    ratio = norm_gray.astype(np.float32) / (gray.astype(np.float32) + 1e-6)
    enhanced = np.clip(bgr.astype(np.float32) * ratio[:, :, np.newaxis], 0, 255)

    hsv = cv2.cvtColor(enhanced.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.4, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0, 255)
    magic = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    show_result(magic)
    set_status("Filter Magic Color diterapkan", SUCCESS_COLOR)


# =====================================
# OCR
# =====================================


def run_ocr():
    if processed_image is None:
        messagebox.showwarning("Peringatan", "Belum ada hasil scan untuk di-OCR")
        return

    if ocr_busy:
        return

    lang_code = LANG_OPTIONS.get(lang_menu.get(), "eng")
    image_snapshot = processed_image.copy()

    btn_ocr.configure(state="disabled")
    set_busy(True, "Menjalankan OCR...")

    def worker():
        fallback_used = False
        try:
            text = pytesseract.image_to_string(
                image_snapshot, lang=lang_code, config="--psm 6"
            )
        except pytesseract.TesseractError:
            fallback_used = True
            try:
                text = pytesseract.image_to_string(
                    image_snapshot, lang="eng", config="--psm 6"
                )
            except Exception as exc:
                app.after(0, lambda exc=exc: on_ocr_error(exc))
                return
        except Exception as exc:
            app.after(0, lambda exc=exc: on_ocr_error(exc))
            return

        app.after(0, lambda: on_ocr_done(text, fallback_used))

    threading.Thread(target=worker, daemon=True).start()


def on_ocr_error(exc):
    btn_ocr.configure(state="normal")
    set_busy(False)
    set_status("OCR gagal dijalankan", ERROR_COLOR)
    messagebox.showerror("Error OCR", f"OCR gagal dijalankan:\n{exc}")


def on_ocr_done(text, fallback_used):
    btn_ocr.configure(state="normal")
    set_busy(False)

    if fallback_used:
        set_status("Bahasa tidak tersedia di Tesseract, OCR memakai English", ERROR_COLOR)
    else:
        set_status("OCR berhasil dijalankan", SUCCESS_COLOR)

    show_ocr_result(text)


def show_ocr_result(text):
    win = ctk.CTkToplevel(app)
    win.title("Hasil OCR")
    win.geometry("600x500")
    win.transient(app)

    textbox = ctk.CTkTextbox(win, wrap="word", font=("Consolas", 13))
    textbox.pack(fill="both", expand=True, padx=15, pady=15)
    textbox.insert("1.0", text if text.strip() else "(Tidak ada teks terdeteksi)")

    btn_row = ctk.CTkFrame(win, fg_color="transparent")
    btn_row.pack(fill="x", padx=15, pady=(0, 15))

    def copy_text():
        app.clipboard_clear()
        app.clipboard_append(textbox.get("1.0", "end-1c"))
        messagebox.showinfo("Disalin", "Teks disalin ke clipboard", parent=win)

    def save_text():
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text File", "*.txt")],
            parent=win,
        )
        if not path:
            return

        with open(path, "w", encoding="utf-8") as file:
            file.write(textbox.get("1.0", "end-1c"))

        messagebox.showinfo("Sukses", "TXT berhasil disimpan", parent=win)

    ctk.CTkButton(btn_row, text="📋 Copy ke Clipboard", command=copy_text).pack(side="left")
    ctk.CTkButton(btn_row, text="💾 Simpan sebagai TXT", command=save_text).pack(
        side="left", padx=10
    )


# =====================================
# MULTI-PAGE DOCUMENT
# =====================================


def add_page():
    if processed_image is None:
        messagebox.showwarning("Peringatan", "Belum ada hasil scan untuk ditambahkan")
        return

    pages.append(processed_image.copy())
    refresh_pages_strip()
    set_status(f"Halaman {len(pages)} ditambahkan ke dokumen", SUCCESS_COLOR)


def remove_page(index):
    if 0 <= index < len(pages):
        pages.pop(index)
        refresh_pages_strip()
        set_status("Halaman dihapus dari dokumen", INFO_COLOR)


def clear_pages():
    if not pages:
        return

    if messagebox.askyesno("Konfirmasi", "Hapus semua halaman dari dokumen?"):
        pages.clear()
        refresh_pages_strip()
        set_status("Semua halaman dihapus", INFO_COLOR)


def refresh_pages_strip():
    for child in pages_strip.winfo_children():
        child.destroy()

    if not pages:
        ctk.CTkLabel(
            pages_strip,
            text="Belum ada halaman ditambahkan. Gunakan 'Tambah Halaman' di sidebar.",
            text_color=INFO_COLOR,
        ).pack(padx=10, pady=10)
        return

    for i, page_img in enumerate(pages):
        card = ctk.CTkFrame(pages_strip, corner_radius=8)
        card.pack(side="left", padx=6, pady=6)

        if len(page_img.shape) == 2:
            disp = page_img
        else:
            disp = cv2.cvtColor(page_img, cv2.COLOR_BGR2RGB)

        thumb = Image.fromarray(disp)
        thumb.thumbnail((80, 100))
        ctk_img = ctk.CTkImage(light_image=thumb, dark_image=thumb, size=thumb.size)

        ctk.CTkLabel(card, image=ctk_img, text="").pack(padx=5, pady=(5, 0))
        ctk.CTkLabel(card, text=f"Hal {i + 1}", font=("Segoe UI", 11)).pack()
        ctk.CTkButton(
            card,
            text="✕",
            width=24,
            height=20,
            fg_color=ERROR_COLOR,
            hover_color="#B91C1C",
            command=lambda idx=i: remove_page(idx),
        ).pack(pady=(0, 5))


def export_multi_pdf():
    if not pages:
        messagebox.showwarning("Peringatan", "Belum ada halaman untuk diexport")
        return

    pdf_path = filedialog.asksaveasfilename(
        defaultextension=".pdf",
        filetypes=[("PDF", "*.pdf")],
    )

    if not pdf_path:
        return

    page_w, page_h = A4
    pdf = canvas.Canvas(pdf_path, pagesize=A4)

    for page_img in pages:
        draw_fitted_image(pdf, page_img, page_w, page_h)
        pdf.showPage()

    pdf.save()

    set_status(f"PDF multi-halaman ({len(pages)} hal) berhasil disimpan", SUCCESS_COLOR)
    messagebox.showinfo("Sukses", "PDF multi-halaman berhasil disimpan")


# =====================================
# SAVE (single page)
# =====================================


def save_jpg():
    if processed_image is None:
        messagebox.showwarning("Peringatan", "Belum ada hasil scan")
        return

    path = filedialog.asksaveasfilename(
        defaultextension=".jpg",
        filetypes=[("JPEG", "*.jpg")],
    )

    if not path:
        return

    cv2.imwrite(path, processed_image)
    set_status("JPG berhasil disimpan", SUCCESS_COLOR)
    messagebox.showinfo("Sukses", "JPG berhasil disimpan")


def save_pdf():
    if processed_image is None:
        messagebox.showwarning("Peringatan", "Belum ada hasil scan")
        return

    pdf_path = filedialog.asksaveasfilename(
        defaultextension=".pdf",
        filetypes=[("PDF", "*.pdf")],
    )

    if not pdf_path:
        return

    page_w, page_h = A4
    pdf = canvas.Canvas(pdf_path, pagesize=A4)
    draw_fitted_image(pdf, processed_image, page_w, page_h)
    pdf.save()

    set_status("PDF berhasil disimpan", SUCCESS_COLOR)
    messagebox.showinfo("Sukses", "PDF berhasil disimpan")


# =====================================
# GUI
# =====================================

if DND_AVAILABLE:
    class App(ctk.CTk, TkinterDnD.DnDWrapper):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.TkdndVersion = TkinterDnD._require(self)
else:
    class App(ctk.CTk):
        pass


app = App()
app.title("Document Scanner Pro")
app.geometry("1600x900")
app.minsize(1280, 750)

# ---- Header ----

header = ctk.CTkFrame(app, fg_color=SIDEBAR_BG, corner_radius=0)
header.pack(fill="x", side="top")

ctk.CTkLabel(
    header, text="📄 DOCUMENT SCANNER PRO", font=("Segoe UI", 28, "bold")
).pack(side="left", padx=30, pady=18)

ctk.CTkLabel(
    header,
    text="Scan  •  Crop  •  Filter  •  OCR  •  Export",
    font=("Segoe UI", 13),
    text_color=INFO_COLOR,
).pack(side="left", pady=18)

ctk.CTkFrame(app, height=3, fg_color=ACCENT, corner_radius=0).pack(fill="x")

# ---- Body: sidebar + main area ----

body = ctk.CTkFrame(app, fg_color="transparent")
body.pack(fill="both", expand=True)

sidebar = ctk.CTkScrollableFrame(body, width=260, fg_color=SIDEBAR_BG, corner_radius=0)
sidebar.pack(side="left", fill="y")

main_area = ctk.CTkFrame(body, fg_color="transparent")
main_area.pack(side="left", fill="both", expand=True, padx=15, pady=15)


def add_section(parent, title_text):
    ctk.CTkLabel(
        parent, text=title_text, font=("Segoe UI", 14, "bold"), anchor="w"
    ).pack(fill="x", padx=15, pady=(18, 6))


# File section
add_section(sidebar, "📂 FILE")
ctk.CTkButton(sidebar, text="📂 Pilih Gambar", command=load_image).pack(
    fill="x", padx=15, pady=4
)
ctk.CTkLabel(
    sidebar,
    text="Atau drag & drop gambar ke jendela" if DND_AVAILABLE else "Drag & drop tidak tersedia",
    font=("Segoe UI", 11),
    text_color=INFO_COLOR,
    wraplength=220,
    justify="left",
).pack(fill="x", padx=15, pady=(0, 4))

# Transform section
add_section(sidebar, "✂️ TRANSFORM")
ctk.CTkButton(sidebar, text="✂️ Auto Scan", command=auto_scan).pack(
    fill="x", padx=15, pady=4
)

btn_manual_mode = ctk.CTkButton(
    sidebar, text="🖱️ Mode Crop Manual", command=toggle_manual_crop
)
btn_manual_mode.pack(fill="x", padx=15, pady=4)
manual_mode_default_color = btn_manual_mode.cget("fg_color")

manual_row = ctk.CTkFrame(sidebar, fg_color="transparent")
manual_row.pack(fill="x", padx=15, pady=(0, 4))
ctk.CTkButton(manual_row, text="↺ Reset", width=110, command=reset_manual_points).pack(
    side="left"
)
btn_apply_manual = ctk.CTkButton(
    manual_row, text="✅ Terapkan", width=110, state="disabled", command=apply_manual_crop
)
btn_apply_manual.pack(side="left", padx=(8, 0))

ctk.CTkButton(sidebar, text="🔄 Rotate", command=rotate_image).pack(
    fill="x", padx=15, pady=4
)
ctk.CTkButton(sidebar, text="📐 Auto Deskew", command=auto_deskew).pack(
    fill="x", padx=15, pady=4
)
ctk.CTkButton(sidebar, text="↩️ Reset", command=reset_image).pack(
    fill="x", padx=15, pady=4
)

# Adjustment section
add_section(sidebar, "🎚️ PENYESUAIAN")

brightness_row = ctk.CTkFrame(sidebar, fg_color="transparent")
brightness_row.pack(fill="x", padx=15)
ctk.CTkLabel(brightness_row, text="Brightness").pack(side="left")
brightness_value_label = ctk.CTkLabel(brightness_row, text="0", text_color=INFO_COLOR)
brightness_value_label.pack(side="right")

brightness_slider = ctk.CTkSlider(sidebar, from_=-100, to=100, command=adjust_brightness)
brightness_slider.set(0)
brightness_slider.pack(fill="x", padx=15, pady=(0, 10))

contrast_row = ctk.CTkFrame(sidebar, fg_color="transparent")
contrast_row.pack(fill="x", padx=15)
ctk.CTkLabel(contrast_row, text="Contrast").pack(side="left")
contrast_value_label = ctk.CTkLabel(contrast_row, text="1.0", text_color=INFO_COLOR)
contrast_value_label.pack(side="right")

contrast_slider = ctk.CTkSlider(sidebar, from_=0.5, to=3.0, command=adjust_contrast)
contrast_slider.set(1)
contrast_slider.pack(fill="x", padx=15, pady=(0, 10))

# Filter section
add_section(sidebar, "🎨 FILTER")
filter_grid = ctk.CTkFrame(sidebar, fg_color="transparent")
filter_grid.pack(fill="x", padx=15, pady=4)
filter_grid.grid_columnconfigure(0, weight=1)
filter_grid.grid_columnconfigure(1, weight=1)

FILTER_BUTTONS = [
    ("⬛ Gray", filter_gray),
    ("◐ B/W", filter_bw),
    ("📊 Histogram", histogram_equalization),
    ("🔲 Edge", edge_detection),
    ("✨ Sharpen", filter_sharpen),
    ("🧹 Denoise", filter_denoise),
    ("🎞️ Sepia", filter_sepia),
    ("🌓 Invert", filter_invert),
    ("🪄 Magic Color", filter_magic_color),
]

for idx, (label, cmd) in enumerate(FILTER_BUTTONS):
    ctk.CTkButton(filter_grid, text=label, command=cmd, width=105).grid(
        row=idx // 2, column=idx % 2, padx=4, pady=4, sticky="ew"
    )

# OCR section
add_section(sidebar, "🔎 OCR")
lang_menu = ctk.CTkOptionMenu(sidebar, values=list(LANG_OPTIONS.keys()))
lang_menu.set("Indonesia + English")
lang_menu.pack(fill="x", padx=15, pady=4)

btn_ocr = ctk.CTkButton(sidebar, text="🔎 Jalankan OCR", command=run_ocr)
btn_ocr.pack(fill="x", padx=15, pady=4)

# Document section
add_section(sidebar, "📑 DOKUMEN")
ctk.CTkButton(sidebar, text="➕ Tambah Halaman", command=add_page).pack(
    fill="x", padx=15, pady=4
)
ctk.CTkButton(
    sidebar,
    text="🗑️ Hapus Semua Halaman",
    command=clear_pages,
    fg_color=ERROR_COLOR,
    hover_color="#B91C1C",
).pack(fill="x", padx=15, pady=4)
ctk.CTkButton(sidebar, text="📑 Export PDF Multi-Halaman", command=export_multi_pdf).pack(
    fill="x", padx=15, pady=4
)

ctk.CTkLabel(
    sidebar, text="Simpan Cepat", font=("Segoe UI", 12, "bold"), text_color=INFO_COLOR
).pack(fill="x", padx=15, pady=(14, 2))

quick_row = ctk.CTkFrame(sidebar, fg_color="transparent")
quick_row.pack(fill="x", padx=15, pady=(0, 20))
ctk.CTkButton(quick_row, text="🖼️ JPG", width=110, command=save_jpg).pack(side="left")
ctk.CTkButton(quick_row, text="📄 PDF", width=110, command=save_pdf).pack(
    side="left", padx=(8, 0)
)

# ---- Main area: image panels ----

panels = ctk.CTkFrame(main_area, fg_color="transparent")
panels.pack(fill="both", expand=True)

left_card = ctk.CTkFrame(panels, fg_color=CARD_BG, corner_radius=14)
left_card.pack(side="left", fill="both", expand=True, padx=(0, 10))

right_card = ctk.CTkFrame(panels, fg_color=CARD_BG, corner_radius=14)
right_card.pack(side="left", fill="both", expand=True, padx=(10, 0))

ctk.CTkLabel(left_card, text="🖼️ Original", font=("Segoe UI", 18, "bold")).pack(
    pady=(15, 10)
)

original_canvas = Canvas(
    left_card, width=DISPLAY_SIZE, height=DISPLAY_SIZE, bg="#11151D", highlightthickness=0
)
original_canvas.pack(expand=True, pady=(0, 20))
original_canvas.bind("<Button-1>", on_canvas_click)
original_canvas.create_text(
    DISPLAY_SIZE // 2,
    DISPLAY_SIZE // 2,
    text="Belum ada gambar\n(drag & drop juga bisa)" if DND_AVAILABLE else "Belum ada gambar",
    fill=INFO_COLOR,
    font=("Segoe UI", 13),
    justify="center",
)

ctk.CTkLabel(right_card, text="✅ Hasil Scan", font=("Segoe UI", 18, "bold")).pack(
    pady=(15, 10)
)
result_label = ctk.CTkLabel(right_card, text="Belum ada hasil", text_color=INFO_COLOR)
result_label.pack(expand=True, pady=(0, 20))

# ---- Pages strip ----

ctk.CTkLabel(
    main_area, text="📑 Halaman Dokumen", font=("Segoe UI", 14, "bold"), anchor="w"
).pack(fill="x", pady=(15, 4))
pages_strip = ctk.CTkScrollableFrame(
    main_area, orientation="horizontal", height=130, fg_color=CARD_BG, corner_radius=12
)
pages_strip.pack(fill="x", pady=(0, 0))
refresh_pages_strip()

# ---- Status bar ----

status_bar = ctk.CTkFrame(app, fg_color=SIDEBAR_BG, corner_radius=0, height=36)
status_bar.pack(fill="x", side="bottom")

status_label = ctk.CTkLabel(status_bar, text="Ready", text_color=INFO_COLOR, anchor="w")
status_label.pack(side="left", padx=20, pady=6)

progress_bar = ctk.CTkProgressBar(status_bar, mode="indeterminate", width=160)

ctk.CTkLabel(
    status_bar, text="UAS Pengolahan Citra - Document Scanner Pro", text_color=INFO_COLOR
).pack(side="right", padx=20, pady=6)

# ---- Drag & drop ----

if DND_AVAILABLE:
    app.drop_target_register(DND_FILES)
    app.dnd_bind("<<Drop>>", handle_drop)

app.mainloop()
