library('gdata')
library('glmnet')
library('grpreg')

tempdata = (read.csv('./Oregon_Met_Data.txt', sep=' ', header=T))
temp_min = sapply(min(tempdata$SID):max(tempdata$SID), function(i) mean(tempdata$Tmin_deg_C[(tempdata$SID == i) & (tempdata$Tmin_deg_C != 9999)]))

locations = read.csv('./Locations.txt', sep=' ', header=T)
coords = locations[,c('Longitude_DD','Latitude_DD')]

spaces = as.matrix(dist(coords, method='euclidean', diag=F, upper=F))
average_dist = mean(spaces)
ranges = sapply(-6:0, function(i){average_dist*1.2^i})

eudist1 = function(x, y) rowSums((coords[x,]-coords[y,])^2)
kernel1 = function(sigma) exp(-outer(1:dim(coords)[1], 1:dim(coords)[1], FUN=eudist1)/(2*sigma^2))
mats1 = do.call('cbind', Map(function(sigma) exp(-spaces^2/(2*sigma^2)), ranges))

model = cv.glmnet(mats1, temp_min, alpha=0)
model = cv.grpreg(mats1, temp_min, group=rep(1:dim(coords)[1], length(ranges)), family='gaussian')
print(model$lambda.min)
print(model$fit$df[model$lambda == model$lambda.min])

longs = seq(min(coords[,1]), max(coords[,1]), length=100)
lats = seq(min(coords[,2]), max(coords[,2]), length=100)

LONGS = outer(longs, lats, function(x, y)x)
LATS = outer(longs, lats, function(x,y)y)
LONGLATS = cbind(c(LONGS), c(LATS))

eudist2 = function(x, y) rowSums((LONGLATS[x,]-coords[y,])^2)
kernel = function(sigma) exp(-outer(1:dim(LONGLATS)[1], 1:dim(coords)[1], FUN=eudist2)/(2*sigma^2))
mats2 = do.call('cbind', Map(kernel, ranges))

pair = predict.cv.glmnet(model, mats2, s='lambda.min')

tempmat = matrix(pair, 100, 100)

scale = max(abs(min(pair)), abs(max(pair)))

image(longs, lats, tempmat, xlab='longtitude', ylab='latitude', useRaster=T)
contour(longs, lats, tempmat, nlevels=10, add=T)
