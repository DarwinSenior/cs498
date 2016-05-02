library(glmnet)

data = t(as.matrix(read.csv('./I2000', header=FALSE, sep=' ')))
tissues = as.matrix(read.csv('./tissue', header=FALSE))
# isTumour represents the tissue type that if it tumour
isTumour = tissues < 0

model = cv.glmnet(data, as.factor(isTumour), family='binomial', type.measure='auc', nfolds=6)
plot(model)
