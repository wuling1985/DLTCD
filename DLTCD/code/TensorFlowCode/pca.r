library(ggfortify)

set.seed(0)
data <- read.csv(file.choose(),  # 資料檔名 
                 header=T,  # 資料中的第一列，作為欄位名稱
                 sep=",")

pca <- prcomp(data)
pca$x
pca$rotation