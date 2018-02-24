library(kernlab)
library(dplyr)

#Load data
setwd("~/Kaggle/MNIST/MNIST-Digit-Recognition/")
mnist_train <- read.csv("./Data/train.csv")
mnist_train$set <- "train"
mnist_train$label <- factor(mnist_train$label)
mnist_train_lbl <- mnist_train$label

mnist_test <- read.csv("./Data/test.csv")
mnist_test$set <- "test"

mnist <- rbind(mnist_train[,2:786], mnist_test)

#PCA
mn_pix <- mnist[, -785]
mn_pca <- prcomp(mn_pix)
#plot(mn_pca, type = "l")
mn_pca <- as.data.frame(mn_pca$x)
mn_pca$set <- mnist$set

mnist_test <- mn_pca[which(mn_pca$set == "test"),1:50]
mnist_train <- mn_pca[which(mn_pca$set == "train"),1:50]
mnist_train$label <- mnist_train_lbl

#Train SVM
mnist_class <- ksvm(label ~ ., data = mnist_train, kernel = "rbfdot")

#Classify test dataset
mnist_pred <- predict(mnist_class, mnist_test)

#Save predictions
result <- data.frame(Label = mnist_pred)
result <- mutate(result, ImageId = rownames(result))
write.csv(result,file = "./Data/prediction.csv", row.names = FALSE)
