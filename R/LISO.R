TT_prep_clean <- read.csv("/media/kap/8e22f6f8-c4df-4d97-a388-0adcae3ec1fb/Python/Thesis/TT_prep_clean.csv", header=FALSE)

y <- TT_prep_clean[, 2]
y <- as.matrix(y)
y <- y-1
x <- TT_prep_clean[, 3:4562]
x <- as.matrix(x)

library("liso", lib.loc="~/R/x86_64-pc-linux-gnu-library/3.2")

trnx = x[1:403,]
trny = y[1:403]
trnx = as.matrix(trnx)
trny = as.matrix(trny)
tstx = x[404:805,]
tsty = y[404:805]
tstx = as.matrix(tstx)
tsty = as.matrix(tsty)
CVobj <- cv.liso(trnx, trny, plot.it = TRUE)
fitobj = liso.backfit(trnx, trny, CVobj$optimlam)

y_hat = fitobj*tstx  # gives the prediction. Need to find a cutoff value
y_m = y_hat >= 0.5
pred_acc = sum((y_m - tsty) == 0)/402  # prediction accuracy is only 52.23%

DataRaw = read.csv("TT_prep_clean.csv")
Yraw = DataRaw[,2]
Xraw = DataRaw[,3:4562]
X = as.matrix(Xraw)
KeepColRaw = read.csv("tt_full.csv")
KeepCol = as.vector(KeepColRaw[,1])
Xred = X[,KeepCol]

shuffleind = sample(seq(1:dim(Xred)[1]))
TrainInd = shuffleind[1:400]
ValInd = shuffleind[401:600]
TestInd = shuffleind[601:800]
XTrain = Xred[TrainInd,]
XVal = Xred[ValInd,]
XTest = Xred[TestInd,]
