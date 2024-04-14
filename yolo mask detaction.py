# Databricks notebook source
dbutils.widgets.text("external_volume_location", "")
dbutils.widgets.text("volume_catalog_name", "")
dbutils.widgets.text("volume_schema_name", "")
dbutils.widgets.text("dataset_location", "")

# COMMAND ----------

external_volume_location = dbutils.widgets.get("external_volume_location")
volume_catalog_name = dbutils.widgets.get("volume_catalog_name")
volume_schema_name = dbutils.widgets.get("volume_schema_name")
dataset_location = dbutils.widgets.get("dataset_location")

# COMMAND ----------

spark.sql(f"""
    CREATE EXTERNAL VOLUME IF NOT EXISTS {volume_catalog_name}.{volume_schema_name}.yolo8
    LOCATION "{external_volume_location}"
    COMMENT "yolo8 poc"
""")

# COMMAND ----------

# MAGIC %pip install ultralytics
# MAGIC
# MAGIC from IPython import display
# MAGIC display.clear_output()
# MAGIC
# MAGIC import ultralytics
# MAGIC ultralytics.checks()

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import torch.distributed as dist
from ultralytics import YOLO

from IPython.display import display, Image

# COMMAND ----------

model = YOLO(f"{dataset_location}/yolov8n.pt")
results = model.predict(source='https://media.roboflow.com/notebooks/examples/dog.jpeg', conf=0.25)

# COMMAND ----------

if not dist.is_initialized():
    dist.init_process_group("nccl")

# Load a model
model = YOLO(f"{dataset_location}/yolov8n.pt")
results = model.train(
    data=f"{dataset_location}/mask/data.yaml", 
    epochs=15,
    project='/tmp/yolo8/', 
    name="yolov8_mask", 
    device=[0]
)

# COMMAND ----------

import cv2
from dbruntime.patches import cv2_imshow

img = cv2.imread("./detect-mask-image.png")

cv2_imshow(img)

# COMMAND ----------

from ultralytics import YOLO
from PIL import Image

model = YOLO("/tmp/yolo8/yolov8_mask/weights/best.pt")
results = model("./detect-mask-image.png") 
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save("./results.jpg")  # save image

# COMMAND ----------

# Load the image via opencv2
img_loc = "./results.jpg"
img = cv2.imread(img_loc)

cv2_imshow(img)

# COMMAND ----------


