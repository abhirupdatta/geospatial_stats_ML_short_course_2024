# Install required packages if not already installed
# install.packages("spBayes")
# install.packages("ggplot2")
# install.packages("sf")
# install.packages("dplyr")

# Load required packages
library(spBayes)
library(ggplot2)
library(sf)
library(dplyr)
library(rworldmap)
library(geoR)
library(MBA)
library(fields)
library(classInt)
library(geoR)
library(RColorBrewer)
library(spNNGP)
library(BRISC)

source("utils.R")

### dataset 1 ###
data1=read.csv("../data/dataset1.csv")


### plot raw data pounts ###

plot(data1$sx,data1$sy,pch=16,xlab="Easting (m)", ylab="Northing (m)",
     col=findColours(classIntervals(data1$y, n=10, style="equal"), col.pal))

plot(data1$sx,data1$sy,pch=16,xlab="Easting (m)", ylab="Northing (m)",
     col=findColours(classIntervals(data1$x, n=10, style="equal"), col.pal))


## plotting surfaces ##

myplot(data1,"y")
myplot(data1,"x")

### linear regression on x
lmobj1=lm(y~x,data=data1)
summary(lmobj1)
data1$res=lmobj1$residuals

myplot(data1,"res",col.br2)


### dataset 2
data2=read.csv("../data/dataset2.csv")

myplot(data2,"y")

lmobj3=lm(y~x,data=data2)


data2$res=lmobj3$residuals

myplot(data2,"res",col.br2)

### empirical variograms ###
max.dist <- 0.75*max(rdist(data1[,1:2]))
bins <- 20

vario1raw <- variog(coords=data1[,1:2], data=data1$y, uvec=(seq(0, max.dist, length=bins)))
plot(vario1raw,pch=16)

vario1 <- variog(coords=data1[,1:2], data=data1$res, uvec=(seq(0, max.dist, length=bins)))
plot(vario1,pch=16)

vario2 <- variog(coords=data2[,1:2], data=data2$y, uvec=(seq(0, max.dist, length=bins)))
plot(vario2,pch=16)

vario2 <- variog(coords=data2[,1:2], data=data2$res, uvec=(seq(0, max.dist, length=bins)))
plot(vario2,pch=16)

### analysis using a Gaussian Process ###
### spatial mle ###

### select training and testing data

set.seed(1234)
index =sample(1:nrow(data2),4*nrow(data2)/5,replace = F)
data2in=data2[index,]
data2out=data2[-index,]

mle <- likfit(coords=data2in[,1:2], data=data2in[,4], trend = trend.spatial(~x,data2in),
  ini.cov.pars=c(0.12,0.2),nugget = 0.02,cov.model="exponential",nospatial=TRUE)

mle

## model comparison ##
mle$AIC
mle$BIC

mle$nospatial$AIC
mle$nospatial$BIC

### out sample predictions (signal = TRUE) ##
sp.pred.obj <- krige.conv(coords=data2in[,1:2], data=data2in[,4],
    locations=data2out[,1:2],krige=krige.control(type.krige="OK",obj.model=mle,
    trend.d=trend.spatial(~x,data2in),trend.l=trend.spatial(~x,data2out)),output=output.control(signal=F))

sp.pred=sp.pred.obj$predict
#data2$res3=data2$y-sp.pred
par(mar=c(4,4,4,4))
plot(data2out$y,sp.pred,xlab="true",ylab="predicted")## scatterplot
abline(a=0,b=1,col="red",lty=2)
#myplot(data2,"res3") ## residual map

rmspe=sqrt(mean((data2out$y-sp.pred)^2))

CI_spatial=sp.pred+1.96*sqrt(sp.pred.obj$krige.var)%*%t(c(-1,1))  ## confidence interval ##
CP_spatial=mean(CI_spatial[,1]<data2out$y & CI_spatial[,2]>data2out$y) ## coverage probability ##

CIW_spatial=mean(CI_spatial[,2]-CI_spatial[,1]) ## confidence interval width ##

CP_spatial
round(CIW_spatial,1)

l3=lm(y~x, data2in)

lm.pred=predict(l3, newdata = data2out, interval = "prediction")

plot(data2out$y,lm.pred[,1],xlab="true",ylab="predicted")## scatterplot
abline(a=0,b=1,col="red",lty=2)

lm.rmspe=sqrt(mean((data2out$y-lm.pred[,1])^2))

lm.CP=mean(lm.pred[,2]<data2out$y & lm.pred[,3]>data2out$y) ## coverage probability ##
lm.CP

lm.CIW=mean(lm.pred[,3]-lm.pred[,2]) ## confidence interval width ##
lm.CIW

round(CIW_spatial,1)

rmspe
lm.rmspe
CP_spatial
lm.CP
CIW_spatial
lm.CIW

############### Plotting Matern covariance function #################################################################

# Define distance range
distances <- seq(0, 5, length.out = 500)

# Define covariance functions
exponential_covariance <- function(d, length_scale = 1.0) {
  exp(-d / length_scale)
}

matern32_covariance <- function(d, length_scale = 1.0) {
  factor <- sqrt(3) * d / length_scale
  (1 + factor) * exp(-factor)
}

gaussian_covariance <- function(d, length_scale = 1.0) {
  exp(-0.5 * (d / length_scale)^2)
}

# Create data frame for plotting
cov_data <- data.frame(
  Distance = rep(distances, 3),
  Covariance = c(exponential_covariance(distances),
                 matern32_covariance(distances),
                 gaussian_covariance(distances)),
  Type = factor(rep(c("Exponential", "Matern 3/2", "Gaussian"),
                    each = length(distances)))
)

# Plot using ggplot2
ggplot(cov_data, aes(x = Distance, y = Covariance, color = Type)) +
  geom_line(size = 1) +
  labs(title = "Covariance Functions as a Function of Distance",
       x = "Distance",
       y = "Covariance") +
  theme_minimal() +
  theme(legend.title = element_blank())





################################################################################
          ###.  House data analysis 1
################################################################################

house=read.csv("../data/housing.csv")

house=na.omit(house) %>% dplyr::filter(median_house_value < 500000)


# Assuming we have longitude and latitude columns


ggplot2::map_data("state") %>%
  filter(region == "california") -> california_map

ggplot() +
  geom_polygon(data = california_map, aes(x = long, y = lat, group = group),
               fill = "lightgray", color = "black") +
  geom_point(data = house, aes(x = longitude, y = latitude, color = median_house_value),
             alpha = 0.6, size = 2) +
  scale_color_viridis_c(option = "plasma", name = "Price (USD)") +
  labs(title = "Housing Data in California",
       x = "Longitude", y = "Latitude") +
  theme_minimal()


l=lm(median_house_value~median_income+
       housing_median_age+total_rooms+total_bedrooms+
       population+households+as.factor(ocean_proximity),
     data=house)

house$res=house$median_house_value-unname(l$fitted.values)


ggplot() +
  geom_polygon(data = california_map, aes(x = long, y = lat, group = group),
               fill = "lightgray", color = "black") +
  geom_point(data = house, aes(x = longitude, y = latitude, color = res),
             alpha = 0.8, size = 2) +
  scale_color_gradient2(low = "red", mid = "white", high = "blue", midpoint = 0, name = "Price") +
  labs(title = "Housing Data in California",
       x = "Longitude", y = "Latitude") +
  theme_minimal()



# Convert the data to a geoR object
geo_data <- as.geodata(house %>% mutate(latitude=jitter(latitude)),
                       coords.col = c("longitude", "latitude"), data.col = "res")

# Create a variogram for the house prices using geoR
variogram_price <- variog(geo_data)

# Plot the variogram
plot(variogram_price, main = "Variogram of House Prices")


###################################################
### WEF data
###################################################
WEF.dat=read.csv("../data/WEFsmall.csv")

WEF.dat$logDBH=log(WEF.dat$DBH)
WEF.dat$Species=as.factor(WEF.dat$Species)

set.seed(1234)
ind=sample(1:nrow(WEF.dat),100,replace=FALSE)

### holdout data to assess RMSPE ###
WEF.out=WEF.dat[ind,]
WEF.in=WEF.dat[-ind,]
rm("WEF.dat")

### diameter at breast height for the trees
logDBH <- WEF.in$logDBH

coords <- as.matrix(WEF.in[,c("East_m","North_m")])

col.br <- colorRampPalette(c("midnightblue", "cyan", "yellow",  "orange", "red"))
col.pal <- col.br(5)

### logDBH quantile based color coding of the locations
quant <- classIntervals(logDBH, n=5, style="quantile")
brks <- round(quant$brks, 2)
quant <- classIntervals(logDBH, n=5, style="fixed",
                        fixedBreaks = brks)

quant.col <- findColours(quant, col.pal)

par(mar=c(4,4,1,1))
plot(coords, col=quant.col, pch=19, cex=1.5, main="", xlab="Easting (m)", ylab="Northing (m)")
legend("topleft", fill=attr(quant.col, "palette"), 
       legend=names(attr(quant.col, "table")), bty="n",cex=1)

### plot of interpolated surface using mba package ###
surf <- mba.surf(cbind(coords,logDBH), no.X=100, no.Y=100, h=5, m=2, extend=FALSE)$xyz.est
image.plot(surf, xaxs = "r", yaxs = "r", xlab="Easting (m)", ylab="Northing (m)", col=col.br(25))

### plot of Species type ###
spnum=as.numeric(WEF.in$Species)
col.pal2 <- col.br(length(unique(spnum)))

plot(coords, col=col.pal2[spnum], pch=19, cex=1.5, main="", xlab="Easting (m)", ylab="Northing (m)")
legend("topleft", fill=col.pal2, 
       legend=levels(WEF.in$Species), bty="n")

### Linear regression ###
lm.logDBH <- lm(logDBH~Species, data=WEF.in)
summary(lm.logDBH)
logDBH.resid <- resid(lm.logDBH)

surf <- mba.surf(cbind(coords,logDBH.resid), no.X=100, no.Y=100, h=5, m=2, extend=FALSE)$xyz.est
image.plot(surf, xaxs = "r", yaxs = "r", xlab="Easting (m)", ylab="Northing (m)", col=col.br2(25))

### variogram of raw data and residuals ###
max.dist=0.5*max(rdist(coords))
bins=20

vario.logDBH <- variog(coords=coords, data=logDBH, uvec=(seq(5, max.dist, length=bins)))
plot(vario.logDBH,pch=16,,ylim=c(0.4,0.6))

vario.logDBH.resid <- variog(coords=coords, data=logDBH.resid, uvec=(seq(0, max.dist, length=bins)))
plot(vario.logDBH.resid,pch=16, ,ylim=c(0.25,0.33))

### spatial mle ###
mle <- likfit(coords=coords, data=logDBH, trend = trend.spatial(~Species,WEF.in), ini.cov.pars=c(0.1,40),
              nugget = 0.25,cov.model="exponential",nospatial=TRUE)

mle

## model comparison ##
round(mle$AIC)
round(mle$BIC)

round(mle$nospatial$AIC)
round(mle$nospatial$BIC)


### in sample predictions (signal = TRUE) ##
sp.pred.obj <- krige.conv(coords=coords, data=logDBH,
  locations=WEF.in[,c("East_m","North_m")],krige=krige.control(type.krige="OK",obj.model=mle,
  trend.d=trend.spatial(~Species,WEF.in),trend.l=trend.spatial(~Species,WEF.in)),output=output.control(signal=TRUE))

sp.pred=sp.pred.obj$predict
logDBH.sp.resid=logDBH-sp.pred

surf <- mba.surf(cbind(coords,logDBH.sp.resid), no.X=100, no.Y=100, h=5, m=2, extend=FALSE)$xyz.est
image.plot(surf, xaxs = "r", yaxs = "r", xlab="Easting (m)", ylab="Northing (m)", col=col.br2(25))

vario.logDBH.sp.resid <- variog(coords=coords, data=logDBH.sp.resid, uvec=(seq(0, max.dist, length=bins)))
plot(vario.logDBH.sp.resid,pch=16)

krig_mlefit=krige.conv(coords=coords, data=logDBH,
  locations=WEF.out[,c("East_m","North_m")],krige=krige.control(type.krige="OK",obj.model=mle,
  trend.d=trend.spatial(~Species,WEF.in),trend.l=trend.spatial(~Species,WEF.out)),output=output.control(signal=F))

pred_spatial=krig_mlefit$predict
rmspe_spatial=sqrt(mean((pred_spatial-WEF.out$logDBH)^2))

pred_lm=as.vector(as.matrix(trend.spatial(~Species,WEF.out))%*%lm.logDBH$coefficients)
rmspe_lm=sqrt(mean((pred_lm-WEF.out$logDBH)^2))

round(rmspe_spatial,2)
round(rmspe_lm,2)


### CP ###
CI_spatial=pred_spatial+1.96*sqrt(krig_mlefit$krige.var)%*%t(c(-1,1))  ## confidence interval ##
CP_spatial=mean(CI_spatial[,1]<WEF.out$logDBH & CI_spatial[,2]>WEF.out$logDBH) ## coverage probability ##
CIW_spatial=mean(CI_spatial[,2]-CI_spatial[,1]) ## confidence interval width ##

CP_spatial
round(CIW_spatial,1)

N=nrow(WEF.out)
#CI_lm=pred_lm+1.96*summary(lm.logDBH)$sigma*cbind(-rep(1,N),rep(1,N))
CI_lm=predict(lm.logDBH,WEF.out,interval="prediction")[,-1]
CP_lm=mean(CI_lm[,1]<WEF.out$logDBH & CI_lm[,2]>WEF.out$logDBH)
CIW_lm=mean(CI_lm[,2]-CI_lm[,1])

CP_lm
round(CIW_lm,1)

### kriged surface ##
## prediction locations ###
WEF.pred=read.csv("../data/WEFpred.csv")

krigsurf_mlefit=krige.conv(coords=coords, data=logDBH,
  locations=WEF.pred[,c("East_m","North_m")],krige=krige.control(type.krige="OK",obj.model=mle,
  trend.d=trend.spatial(~Species,WEF.in),trend.l=trend.spatial(~Species,WEF.pred)),output=output.control(signal=F))

pred=krigsurf_mlefit$predict
predsd=sqrt(krigsurf_mlefit$krige.var)

predsurf <- mba.surf(cbind(WEF.pred[,c("East_m","North_m")],pred), no.X=100, no.Y=100, h=5, m=2, extend=FALSE)$xyz.est
image.plot(predsurf, xaxs = "r", yaxs = "r", xlab="Easting (m)", ylab="Northing (m)", col=col.br(25))

predsdsurf <- mba.surf(cbind(WEF.pred[,c("East_m","North_m")],predsd), no.X=100, no.Y=100, h=5, m=2, extend=FALSE)$xyz.est
image.plot(predsdsurf, xaxs = "r", yaxs = "r", xlab="Easting (m)", ylab="Northing (m)", col=rev(terrain.colors(25)))

image.plot(predsdsurf, xaxs = "r", yaxs = "r", xlab="Easting (m)", ylab="Northing (m)", col=rev(terrain.colors(25)))
points(WEF.pred[,1:2],col="black",pch=16,cex=0.5)
points(WEF.pred[which(WEF.pred$Species=="GF"),1:2],col="cyan",pch=16,cex=1.2)


##### BCEF data ###
data(BCEF)

BCEF %>% ggplot() +
  geom_point(aes(x=x,y=y,col=FCH)) + 
  scale_color_viridis_c(option = "plasma") +
  theme_minimal()


BCEF %>% ggplot() + 
  geom_point(aes(x=x,y=y,col=PTC)) + 
  scale_color_viridis_c() +
  theme_minimal()

# BCEF %>% ggplot() +
#   geom_point(aes(x=PTC,y=FCH),alpha=0.1)
#   theme_minimal()
#   
# l=lm(FCH ~ PTC, data=BCEF)  
# l.pred=predict(l,newdata=BCEF)
# 
# plot(BCEF$PTC,BCEF$FCH)
# points(BCEF$PTC,l.pred,col="red")
# BCEF %>% ggplot() + 
#   geom_point(aes(x=x,y=y,col=as.factor(holdout))) + 
#   scale_color_discrete(name = "Holdout") +
#   theme_minimal()


l.bcef=lm(FCH~PTC,data=BCEF)
BCEF$res=BCEF$FCH-l.bcef$fitted.values


BCEF %>% ggplot() +
  geom_point(aes(x=x,y=y,col=res)) + 
  scale_color_gradient2(low = "red", mid = "white", high = "blue", midpoint = 0, name = "") +
  theme_minimal()


#l2.bcef=lm(FCH~PTC,data=BCEF)


# Convert the data to a geoR object

set.seed(1234)
subindex=sample(1:nrow(BCEF),10000)
geo_data <- as.geodata(BCEF[subindex,],
    coords.col = c("x", "y"), data.col = "res")

# Create a variogram for the house prices using geoR
max.dist <- 0.75*max(rdist(BCEF[subindex,c("x", "y")]))
bins <- 20

# Plot the variogram
vario1res <- variog(geo_data, uvec=(seq(0, max.dist, length=bins)))
plot(vario1res,pch=16)

# t0=Sys.time()
# br <- BRISC_estimation(unname(as.matrix(BCEF.in[,c("x","y")])),
#           sqrt(BCEF.in$FCH), cbind(1,BCEF.in$PTC),n.neighbors = 5)
# t1=Sys.time()
# print(t1-t0)


### running BRISC (checking results with likfit for small data)
set.seed(1234)
index=sample(1:nrow(BCEF),1000)

BCEF.sub=BCEF[index,]

t0=Sys.time()
br <- BRISC_estimation(unname(as.matrix(BCEF.sub[,c("x","y")])),
        BCEF.sub$FCH, cbind(1,BCEF.sub$PTC),n.neighbors = 15)
t1=Sys.time()
print(t1-t0)

mle <- likfit(coords=unname(as.matrix(BCEF.sub[,c("x","y")])), data=BCEF.sub$FCH, trend = trend.spatial(~PTC,BCEF.sub),
    ini.cov.pars=c(0.12,0.2),nugget = 0.02,cov.model="exponential",nospatial=TRUE)

br$Beta
br$Theta

mle

#################################################################

set.seed(1234)
index=sample(1:nrow(BCEF),100000)
BCEF.in=BCEF[index,]
BCEF.out=BCEF[-index,]

BCEF.in %>% ggplot() + 
  geom_point(aes(x=x,y=y,col=PTC)) + 
  scale_color_viridis_c() +
  theme_minimal()

### WARNING !! This run takes about 2 hours (it is analyzing a spatial dataset of size 100000)
t0=Sys.time()
br <- BRISC_estimation(unname(as.matrix(BCEF.in[,c("x","y")])),
  BCEF.in$FCH, cbind(1,BCEF.in$PTC),n.neighbors = 5, tau.sq = 12, sigma.sq = 30, phi =2)
t1=Sys.time()
print(t1-t0)
save(file="bcef_BRISC_fit.Rdata",br)
### end of warning ###

### loading the pre-saved BRISC model fit R object
load("bcef_BRISC_fit.Rdata")

br$n.neighbors=15 ## increasing the neighbor size for predictions

## this takes about 2 minutes
br.pred <- BRISC_prediction(br,unname(as.matrix(BCEF.out[,c("x","y")])),cbind(1,BCEF.out$PTC))

plot(BCEF.out$FCH,br.pred$prediction,ylab="Prediction",xlab="True")
abline(a=0,b=1,col="red")
BCEF.out$res=BCEF.out$FCH-br.pred$prediction

BCEF.out %>% ggplot() +
  geom_point(aes(x=x,y=y,col=res)) + 
  scale_color_gradient2(low = "red", mid = "white", high = "blue", midpoint = 0, name = "") +
  theme_minimal()

sp.rmspe=sqrt(mean(BCEF.out$res^2))
sp.rmspe

sp.CP=mean(br.pred$prediction.ci[,1]<BCEF.out$FCH & br.pred$prediction.ci[,2]>BCEF.out$FCH) ## coverage probability ##
sp.CP

sp.CIW=mean(br.pred$prediction.ci[,2]-br.pred$prediction.ci[,1]) ## confidence interval width ##
sp.CIW

l.bcef=lm(FCH~PTC,data=BCEF.in)
BCEF.in$res=BCEF.in$FCH-l.bcef$fitted.values

lm.pred=predict(l.bcef, newdata = BCEF.out, interval = "prediction")

plot(BCEF.out$FCH,lm.pred[,1],ylab="Prediction",xlab="True")## scatterplot
abline(a=0,b=1,col="red")

lm.rmspe=sqrt(mean((BCEF.out$FCH-lm.pred[,1])^2))
lm.rmspe

lm.CP=mean(lm.pred[,2]<BCEF.out$FCH & lm.pred[,3]>BCEF.out$FCH) ## coverage probability ##
lm.CP

lm.CIW=mean(lm.pred[,3]-lm.pred[,2]) ## confidence interval width ##
lm.CIW
