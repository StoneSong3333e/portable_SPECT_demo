# Motion + Closed-Loop Control (Rotary + Linear)

This repository provides two complementary Python modules for precision stage control in laboratory and medical imaging systems.

- **Motion Control** (`src/motion_control/velmex_control.py`):  
  Core driver for a Velmex VXM-based system controlling a rotary stage (B48Stage72) and a linear axis (MA60).

- **Closed-Loop Trim Controller** (`src/closed_loop/detector_closed_loop.py`):  
  Detector-agnostic feedback module that corrects both **angle** and **radius** through real-time sensing until within tolerance.

---

## 🧩 Features
- Modular design suitable for **high-precision control stages**
- Flexible combinations of **linear, rotational, or hybrid axes**
- Detector interface is fully **open and replaceable**:
  - Hardware sensors (encoders, laser probes, etc.)
  - **Image-based feedback** — use processed imaging data to infer detector displacement  
    by matching feature points between frames, eliminating hardware sensors while improving precision
- Proportional correction with bounded trim steps and dropout handling
- Angle wrap-around correction in (-180°, 180°]

---

## 🧠 Applications
Originally developed for **adaptive medical imaging**, the framework generalizes to any  
experiment requiring sub-millimeter accuracy and adaptive correction.

It can be directly extended with image-processing and geometric-inference modules  
to reconstruct motion from visual data — merging control theory and computer vision.

---

## 🤝 Contribution
Contributions are welcome, especially in:
- Image-based detector feedback  
- Predictive error compensation  
- Lightweight AI-assisted motion inference  

Fork this repo and help us advance precision imaging control.

---

## ⚙️ License
MIT License.
