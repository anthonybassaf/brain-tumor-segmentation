import os
dir = "./dataset/kaggle_3m"

# To call the corresponding original file
def tiff_call(content):
  content = content.split(".")[0] + ".tif"
  dir_ = "_".join(content.split("_")[:4])
  path = dir + "/" + dir_ + "/" + content
  if os.path.exists(path):
    return path
  else :
    return "lgg-mri-segmentation/kaggle_3m/TCGA_CS_5395_19981004/TCGA_CS_5395_19981004_12.tif"