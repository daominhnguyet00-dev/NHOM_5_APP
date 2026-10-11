# 📈 HOSE QUANT TRADING WEB APP — CHIẾN LƯỢC KẾT HỢP SMA VÀ OBV

Ứng dụng web định lượng chuyên nghiệp được xây dựng trên nền tảng **Streamlit**, dùng để kiểm định (backtest) và tối ưu hóa hiệu quả của chiến lược giao dịch kết hợp giữa **Đường trung bình động giản đơn (SMA)** và **Chỉ báo khối lượng cân bằng (OBV)** trên dữ liệu cổ phiếu sàn **HOSE (2020 – 2023)**.

---

## 🌟 1. Giới Thiệu & Điểm Nổi Bật

Trong thị trường chứng khoán Việt Nam (HOSE), việc kết hợp giữa chỉ báo **xu hướng giá (SMA)** và chỉ báo **dòng tiền khối lượng (OBV)** giúp nhận diện sớm các điểm đảo chiều và bảo toàn vốn vượt trội:

- **Loại bỏ Look-ahead Bias:** Tín hiệu kỹ thuật phát sinh ở phiên $t$ chỉ thực thi lệnh tại phiên kế tiếp $t+1$ (`shift(1)`).
- **Điều chỉnh giá (Adjusted Price):** Tự động ưu tiên `adj_close` để hạn chế méo mó do chia tách cổ tức và quyền mua.
- **Sàng lọc Đa tiêu chí (TA Screening):** Đánh giá 4 trụ cột kỹ thuật (*Trend, Momentum, Volume, Risk*) trên tập Train (2020–2021) để chọn Top 5 cổ phiếu tiềm năng (`DIG`, `DPM`, `NLG`, `KBC`, `DCM`).
- **Warm-up Lịch sử đầy đủ:** Kiểm định ngoài mẫu (Out-of-sample Test 2022) sử dụng toàn bộ dữ liệu trước đó để làm ấm (warm-up) các đường SMA dài hạn (lên tới SMA350).
- **Phân bổ danh mục MPT (Markowitz):** Tối ưu hóa Sharpe Ratio trên ma trận lợi nhuận của 5 luồng chiến lược với ràng buộc tỷ trọng tập trung $[5\%, 40\%]$.
- **Benchmark Buy & Hold Chuẩn Xác:** Không tái cân bằng (rebalance) thụ động hàng ngày, phản ánh đúng thực tế đầu tư mua và nắm giữ.

---

## 📊 2. Kết Quả Kiểm Định Tiêu Biểu (Thị Trường Gấu 2022)

Trong năm 2022 khi chỉ số VN-Index sụt giảm mạnh:

| Phương pháp | Lợi nhuận tổng thể | Tỷ suất năm | Biến động năm | Sharpe Ratio | Max Drawdown |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Equal Weight Buy & Hold** (Benchmark) | **-42.27%** | -42.66% | 46.40% | -0.964 | **-60.52%** |
| **SMA+OBV OR Equal Weight** | **-8.63%** | -8.73% | 26.29% | -0.217 | **-31.85%** |
| **SMA+OBV OR MPT (Tối ưu)** | **-3.21%** | **-3.25%** | 28.06% | **+0.022** | **-29.22%** |

> **Nhận xét:** Chiến lược kết hợp SMA + OBV giúp giảm mức sụt giảm tối đa từ **-60.52%** xuống còn **-29.22%**, và danh mục MPT duy trì được chỉ số Sharpe dương (**+0.022**) ngay trong thị trường gấu khốc liệt.

---

## 📁 3. Cấu Trúc Thư Mục Dự Án

```text
NHOM_5_APP/
├── app.py                      # Mã nguồn chính của ứng dụng Streamlit
├── requirements.txt            # Danh sách thư viện Python phụ thuộc
├── README.md                   # Hướng dẫn chi tiết dự án & triển khai
├── HOSE_2020_2023_in.csv       # File dữ liệu giao dịch lịch sử HOSE (2020 - 2023)
└── HOSE_STRATEGY_8_5_REVISED.ipynb # Notebook nghiên cứu định lượng gốc
```

---

## 🚀 4. Hướng Dẫn Cài Đặt & Chạy Trực Tiếp Tại Local

### Bước 1: Chuẩn bị môi trường Python
Khuyến nghị sử dụng Python **3.9**, **3.10** hoặc **3.11**.

### Bước 2: Clone hoặc mở thư mục dự án
Mở Terminal hoặc PowerShell tại thư mục `NHOM_5_APP`:
```bash
cd NHOM_5_APP
```

### Bước 3: Tạo môi trường ảo (Khuyến nghị)
```bash
# Trên Windows:
python -m venv venv
.\venv\Scripts\activate

# Trên macOS/Linux:
python3 -m venv venv
source venv/bin/activate
```

### Bước 4: Cài đặt các thư viện cần thiết
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Bước 5: Khởi chạy ứng dụng Web
```bash
streamlit run app.py
```
Trình duyệt sẽ tự động mở ứng dụng tại địa chỉ: `http://localhost:8501`.

---

## 🌐 5. Hướng Dẫn Tải Lên GitHub & Deploy Lên Streamlit Cloud

### Bước 1: Đẩy mã nguồn lên GitHub
1. Đăng nhập tài khoản trên [GitHub](https://github.com/) và tạo một **New Repository** (ví dụ: `hose-trading-strategy-app`, để chế độ **Public**).
2. Mở Terminal / PowerShell tại thư mục dự án và chạy các lệnh:
```bash
git init
git add app.py requirements.txt README.md HOSE_2020_2023_in.csv
git commit -m "Initial commit: Streamlit app for HOSE SMA+OBV trading strategy"
git branch -M main
git remote add origin https://github.com/<TEN_GITHUB_CUA_BAN>/<TEN_REPO>.git
git push -u origin main
```
*(Thay `<TEN_GITHUB_CUA_BAN>` và `<TEN_REPO>` bằng thông tin GitHub của bạn)*

### Bước 2: Deploy lên Streamlit Community Cloud (Miễn phí)
1. Truy cập vào trang web [share.streamlit.io](https://share.streamlit.io/) và đăng nhập bằng tài khoản GitHub của bạn.
2. Nhấn vào nút **"Create app"** (hoặc **"New app"**).
3. Điền thông tin cấu hình triển khai:
   - **Repository:** Chọn repository bạn vừa tải lên (ví dụ: `your-username/hose-trading-strategy-app`).
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. (Tùy chọn) Nhấn vào **Advanced settings** nếu muốn chọn phiên bản Python (khuyến nghị Python 3.10 hoặc 3.11).
5. Nhấn nút **"Deploy!"**.
6. Chờ trong khoảng 1–2 phút để hệ thống tự động cài đặt các gói trong `requirements.txt` và khởi chạy. Ứng dụng sẽ có link công khai dạng:
   `https://<ten-app>.streamlit.app` để bạn chia sẻ cho mọi người!

---

## 🛠️ 6. Các Tính Năng Chính Trên Giao Diện Web App

1. **📊 Tab 1: Tổng quan Dữ liệu:**
   - Xem thống kê số lượng mã, số phiên giao dịch, cấu trúc cột dữ liệu.
   - Bộ lọc xem trước dữ liệu giá OHLCV và biểu đồ kỹ thuật của từng mã.
   - Hỗ trợ tải file CSV mới từ máy tính hoặc sử dụng dữ liệu mẫu có sẵn.
2. **🎯 Tab 2: Sàng lọc Cổ phiếu (TA Screening):**
   - Tự do tùy biến trọng số 4 trụ cột hoặc chọn các kịch bản chuẩn (*Base*, *Equal*, *Risk-oriented*).
   - Tự động xếp hạng và xuất bảng Top N cổ phiếu dẫn đầu kèm Radar Chart so sánh.
3. **📈 Tab 3: Chiến lược Từng Cổ phiếu:**
   - Biểu đồ tương tác đa tầng (Subplots) hiển thị đồng thời: Giá & SMA Crossover, Điểm mua/bán (Triangles), Chỉ báo OBV & MA, và Trạng thái vị thế nắm giữ (Long / Cash).
   - Bảng đo lường hiệu suất cá nhân so với phương thức Mua & Nắm giữ (Buy & Hold).
4. **💼 Tab 4: Danh mục MPT vs Equal Weight vs Buy & Hold:**
   - Biểu đồ tăng trưởng tài sản (Equity Curve) trực quan so sánh 3 đường lợi nhuận.
   - Biểu đồ sụt giảm tài sản (Underwater / Drawdown Plot).
   - Bảng phân bổ tỷ trọng MPT Markowitz (Pie Chart) và kết luận tự động từ mô hình.
5. **🔬 Tab 5: Phân tích Độ nhạy (Sensitivity):**
   - Đánh giá khả năng chống chịu của danh mục theo 4 kịch bản phí giao dịch ($0.0\%$, $0.05\%$, $0.10\%$, $0.20\%$).
6. **📖 Tab 6: Cơ sở Phương pháp luận:**
   - Trình bày chi tiết cơ sở toán học, công thức tính toán và các nguyên tắc loại bỏ sai lệch kiểm định.

---

## 👥 Nhóm Thực Hiện & Bản Quyền
- **Nhóm nghiên cứu:** Nhóm 5 — Phân tích Định lượng & Kiểm định Chiến lược Đầu tư Sàn HOSE
- **Nền tảng công nghệ:** Python, Streamlit, Plotly, Scipy Optimization, TA-Lib/TA Library.
