library(class)
library(dplyr)
#Load data
setwd("~/Kaggle/MNIST/MNIST-Digit-Recognition/")
mnist_train <- read.csv("./Data/train.csv")
mnist_train$label <- factor(mnist_train$label)

#PCA
mn_pix <- mnist_train[, -1]
mn_label <- mnist_train$label
mn_pca <- prcomp(mn_pix)
mnist_train <- as.data.frame(mn_pca$x)
mnist_train <- mnist_train[, 1:50]

#Classify test dataset
mnist_test <- read.csv("./Data/test.csv")
mnist_test <- predict(mn_pca, mnist_test)
mnist_test <- as.data.frame(mnist_test)
mnist_test <- mnist_test[, 1:50]
mnist_pred <- knn(train = mnist_train, test = mnist_test, cl = mn_label, k = 205)

#Save predictions
result <- data.frame(Label = mnist_pred)
result <- mutate(result, ImageId = rownames(result))
write.csv(result,file = "./Data/prediction.csv", row.names = FALSE)
