library(dplyr)
library(neuralnet)
#Load data
setwd("~/Kaggle/MNIST/MNIST-Digit-Recognition/")
mnist <- read.csv("./Data/train.csv")

#PCA
mn_pix <- mnist[, -1]
mn_label <- mnist$label
mn_pca <- prcomp(mn_pix)
plot(mn_pca, type = "l")
mnist <- as.data.frame(mn_pca$x)
mnist <- mnist[, 1:10]
mnist$label <- mn_label

# Binarize the categorical output
for(i in c(0:9)) {
  mnist <- cbind(mnist, mnist$label == i)
}

digits <- c('i0', 'i1', 'i2','i3','i4','i5','i6','i7','i8','i9')
names(mnist)[12:21] <- digits
mnist[12:21] <- sapply(mnist[12:21], as.numeric)

#Split dataset
mnist_train <- mnist[1:2000, ]
mnist_train_lbl <- mnist_train$label
mnist_test <- mnist[21001:42000, ]
mnist_test_lbl <- mnist_test$label

#Neural net
f <- paste(paste(digits, collapse = " + "), " ~ ", paste(names(mnist_train[1:10]), collapse = " + "))
mnist_model <- neuralnet(f, data = mnist_train,hidden = 70)
#plot(mnist_model)
model_result <- compute(mnist_model, mnist_test[1:10])
predicted <- as.data.frame(model_result$net.result)
names(predicted) <- digits
predicted$label <- colnames(predicted)[apply(predicted,1,which.max)]
prediction <- sapply(substr(predicted$label,2,2), as.numeric)

result <- data.frame(prediction,mnist_test_lbl)
result$check <- ifelse(result$prediction == result$mnist_test_lbl,"1","0")
table(result$check)
table(result$check,result$mnist_test_lbl)


