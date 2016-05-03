library("glmnet")
library("AUC")


data = t(as.matrix(read.csv('./I2000', header=FALSE, sep=' ')))
tissues = as.matrix(read.csv('./tissue', header=FALSE))
# isTumour represents the tissue type that if it tumour
isTumour = tissues < 0

avg_auc = 0
avg_deviance = 0
avg_df = 0
for (i in 1:10){
    model = cv.glmnet(data, isTumour, family='binomial', type.measure='auc', nfolds=6)
    plot(model)
    avg_auc = avg_auc + max(model$cvm)
    best_model = glmnet(data, isTumour, family='binomial', lambda=model$lambda.min)
    avg_deviance = avg_deviance + deviance(best_model)
    avg_df = avg_df + best_model$df
}

print(avg_auc/10)
print(avg_deviance/10)
print(avg_df/10)
