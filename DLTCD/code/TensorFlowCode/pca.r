library(ggfortify)

set.seed(0)
data <- read.csv(file.choose(),  # ����ɦW 
                 header=T,  # ��Ƥ����Ĥ@�C�A�@�����W��
                 sep=",")

pca <- prcomp(data)
pca$x
pca$rotation