#Load data
setwd("~/Kaggle/MNIST/MNIST-Digit-Recognition/")
mnist <- read.csv("./Data/train.csv")
mnist$label <- factor(mnist$label)
table(mnist$label)

# PCA
mn_pix <- mnist[, -1]
mn_label <- mnist$label
mn_pca <- prcomp(mn_pix)
plot(mn_pca, type = "l")
summary(mn_pca)
mnist <- as.data.frame(mn_pca$x)
mnist <- mnist[, 1:50]
mnist$label <- mn_label

#Split dataset
mnist_train <- mnist[1:21000, ]
table(mnist_train$label)
mnist_train_lbl <- mnist_train$label
mnist_test <- mnist[21001:42000, ]
table(mnist_test$label)
mnist_test_lbl <- mnist_test$label

#Validate model
library(class)
mnist_pred <- knn(train = mnist_train, test = mnist_test, cl = mnist_train_lbl, k =145) # sqrt(21000) = 145 

#Evaluate model
library(gmodels)
CrossTable(x = mnist_test_lbl, y=mnist_pred, prop.c=FALSE,prop.t=FALSE)


