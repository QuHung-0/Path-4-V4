# Path 4 - Fish Species Classification + Neo4j Knowledge Graph

## Mô tả đề tài
Hệ thống nhận dạng loài cá sử dụng mô hình phân loại ảnh ResNet-18 kết hợp
chuẩn hóa tên loài qua WoRMS API và lưu trữ kết quả vào Knowledge Graph Neo4j.
Giao diện web Flask cho phép người dùng upload ảnh, nhận kết quả phân loại và
tra cứu lịch sử dự đoán.

## Yêu cầu hệ thống
- Python 3.10
- NVIDIA GPU có CUDA 11.8 (hoặc CPU nếu không có GPU)
- Neo4j Desktop >= 2026.x đang chạy trên bolt://127.0.0.1:7687
- Kết nối internet (chỉ cần khi chạy M05_P02 lần đầu để lấy taxonomy từ WoRMS)
- Pycharm 
## Cài đặt

```bash
# Cài đặt thư viện trong pycharm terminal
pip install -r requirements.txt
```

```
D:\ASDF\Nam 4 HK 2\Đồ án tốt nghiệp\Path 4 V4\
.gitattributes
.idea
.venv
config
dataset
kg
model_weights
outputs
report
requirements.txt
semantic
webapp
z. image

```

## Cấu hình trước khi chạy:
##### 1. Vào file Path 4 V4\config\M00_pipeline_config\M00_C01_allPipelinePaths.py


```

# ─── ROOT ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = r"D:\ASDF\Nam 4 HK 2\Đồ án tốt nghiệp\Path 4 V4"

PROJECT_ROOT_OUTPUTS = r"D:\ASDF\Nam 4 HK 2\Đồ án tốt nghiệp\Path 4 V4\outputs"


# ─── M01 DATASET ─────────────────────────────────────────────────────────────

M01_INPUT_RAW_FISH_IMAGE_FOLDER = r"D:\ASDF\Nam 4 HK 2\Đồ án tốt nghiệp\Path 4 V4\dataset\DATA DOWNLOAD\fish_image"

```

Thay đổi phù hợp với đường dẫn hệ thống hiện thỉ trong thư mục để các files chạy đúng với cài đặt, hiện tại cài đặt đang ở trong thư mục: "D:\ASDF\Nam 4 HK 2\Đồ án tốt nghiệp\Path 4 V4" nên sẽ để theo cài đặt như trên,

nếu files ở vị trí: "C:\Users\ACER\Desktop\Path 4 V4" thì được cấu hình thành:

```

# ─── ROOT ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = r"C:\Users\ACER\Desktop\Path 4 V4"

PROJECT_ROOT_OUTPUTS = r"C:\Users\ACER\Desktop\Path 4 V4"


# ─── M01 DATASET ─────────────────────────────────────────────────────────────

M01_INPUT_RAW_FISH_IMAGE_FOLDER = r"C:\Users\ACER\Desktop\Path 4 V4\dataset\DATA DOWNLOAD\fish_image"

```

##### 2. Vào thư mục Path 4 V4\config\M00_pipeline_config\M00_C02_allSettings.py

```
M06_P01_NEO4J_URI      = "neo4j://127.0.0.1:7687"
M06_P01_NEO4J_USER     = "neo4j"
M06_P01_NEO4J_PASSWORD = "123456789"
M06_NEO4J_DATABASE = "neo4j"
```

Để mật khẩu (M06_P01_NEO4J_PASSWORD = "123456789") trùng với password trong neo4j khi tạo instance:
![image description](https://github.com/QuHung-0/Path-4-V4/blob/main/z.%20image/Neo4j%20instance.png)
## Thứ tự chạy (lần đầu tiên)
Mở trong pycharm: click chuột phải vào files:
![image description](https://github.com/QuHung-0/Path-4-V4/blob/main/z.%20image/Run%20Py.png)

và chạy theo thứ tự
```
M01: Path 4 V4\dataset\M01_dataset\process:
P01 - P02 - P03 - P04

M02: Path 4 V4\dataset\M02_preprocessing\process:
P01 - P02

M03: Path 4 V4\model_weights\M03_model\process:
P01 - P02 - P03 - P05 - P07 - P06

M04: Path 4 V4\model_weights\M04_evaluation\process:
P01 - P02 - P03 - P04 - P05

M05: Path 4 V4\semantic\M05_semantic\process:
P01 - P02 - P03 - P05 - P06

M06: Path 4 V4\kg\M06_knowledge_graph\process (phải có neo4j chạy cùng):
P01 - P03 - P04

M07: Path 4 V4\webapp\M07_web_app\process:
P05
```
Nếu muốn reset toàn bộ hệ thống để chạy hoàn toàn mới:
xóa thư mục: Path 4 V4\outputs