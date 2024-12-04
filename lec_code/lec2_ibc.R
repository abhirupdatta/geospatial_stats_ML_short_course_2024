# Install required packages if not already installed

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
library(randomForest)
library(mgcv)
library(RandomForestsGLS)
library(ggmap)
library(ggspatial)
library(sf)
library(prettymapr)
library(osmdata)
library(mvtnorm)
library(parallel)
library(tidyverse)
library(scales)
library(spatialRF). ## install spatialRF using this command remotes::install_github(repo = "blasbenito/spatialRF", ref = "main",force = TRUE,quiet = TRUE)
numCores <- detectCores()

source("utils.R")

### CART vs DART split criterion ###
mu.gen=function(x) 5*(x<0.5) + 7.5*(x>=0.5)
plot(seq(0,1,0.01),mu.gen(seq(0,1,0.01)),xlab="x",ylab="m(x)",pch=16,ylim=c(0,10))

split.compare=function(seed){
set.seed(seed)
print(seed)
n=100
x=runif(n,0,1)
mu=mu.gen(x)

coords <- cbind(runif(n,0,1), runif(n,0,1))
sigma.sq = 5
phi = 2
tau.sq = 0.1
D <- as.matrix(dist(coords))
R <- exp(-phi*D)
w  <- rmvn(1, rep(0,n), sigma.sq*R)

y <- rnorm(n, mu.gen(x) + w, sqrt(tau.sq))

cvec=seq(0.1,0.9,0.05)

cart=function(c,y,x){ 
  ind=which(x<c)
  var(y[ind]) + var(y[-ind])
}


dart=function(c,y,x){ 
  Z=cbind(1*(x<c),1*(x>=c))
  v=solve(R,Z)
  beta=solve(t(v)%*%Z,t(v)%*%y)
  u=y-as.vector(Z%*%beta)
  mean(u*u)
}

df=rbind(data.frame(seed=seed,cutoff=cvec,method='CART',loss=sapply(cvec,cart,y,x)),
         data.frame(seed=seed,cutoff=cvec,method='DART',loss=sapply(cvec,dart,y,x)))
df
}

results.split.compare=Reduce('rbind',lapply(1:100,split.compare))

summary.split.compare=results.split.compare %>% 
  group_by(method,cutoff) %>%
  summarise(split.criterion=mean(loss),low=quantile(loss,0.025),high=quantile(loss,0.975))

summary.split.compare %>%
  ggplot() +
  geom_ribbon(aes(x=cutoff, ymin = low, ymax = high, fill = method),alpha=0.5) +
  scale_fill_manual(values = c("#F8766D","maroon")) +
  geom_line(aes(x=cutoff, y = split.criterion,col=method),size=2) +
  scale_color_manual(values = c("#F8766D","maroon")) +
  facet_wrap(. ~ method) +
  theme(legend.title=element_blank(),text = element_text(size = 20))+
  ylab("Split function")

results.split.compare %>% 
  group_by(method,seed) %>%
  summarise(c_hat=cutoff[which.min(loss)]) %>%
  ggplot() +
  geom_density(aes(x=c_hat,col=method),size=1.2) +
  theme(legend.title=element_blank(),text = element_text(size = 20)) +
  xlab("Selected cutoff") +
  scale_color_manual(values = c("#F8766D","maroon"))
  
  
#### example of RF doing poorly for correlated data ####

## first iid case ##
set.seed(5)
n <- 200
coords <- cbind(runif(n,0,1), runif(n,0,1))
set.seed(2)
x <- as.matrix(runif(n),n,1)
sigma.sq = 10
phi = 1
tau.sq = 0.1
D <- as.matrix(dist(coords))
R <- exp(-phi*D)
w  <- rmvn(1, rep(0,n), sigma.sq*R)
# y <- rnorm(n, 10*sin(pi * x) + w, sqrt(tau.sq))

#w  <- rmvn(1, rep(0,n), sigma.sq*R)
y <- rnorm(n, 10*sin(pi * x), sqrt(tau.sq+var(w)))

Xtest <- matrix(seq(0,1, by = 1/10000), 10001, 1)


set.seed(1)
RF_est.iid <- randomForest(x, y, nodesize = 20, ntree=50)
RF_predict.iid <- predict(RF_est.iid, Xtest)

rf_loess_10.iid <- loess(RF_predict.iid ~ c(1:length(RF_predict.iid)), span=0.1)
rf_smoothed10.iid <- predict(rf_loess_10.iid)

df=data.frame(sx=coords[,1],sy=coords[,2],x=x,y=y,mu=10*sin(pi * x),error=y -10*sin(pi * x))
myplot(df,"error",col.br2)

#### now the spatially correlated case

set.seed(5)
n <- 200
coords <- cbind(runif(n,0,1), runif(n,0,1))
set.seed(2)
x <- as.matrix(runif(n),n,1)
sigma.sq = 10
phi = 1
tau.sq = 0.1
D <- as.matrix(dist(coords))
R <- exp(-phi*D)
w  <- rmvn(1, rep(0,n), sigma.sq*R)
y <- rnorm(n, 10*sin(pi * x) + w, sqrt(tau.sq))

# w  <- rmvn(1, rep(0,n), sigma.sq*R)
# y <- rnorm(n, 10*sin(pi * x), sqrt(tau.sq+var(w)))

Xtest <- matrix(seq(0,1, by = 1/10000), 10001, 1)

set.seed(1)
RF_est <- randomForest(x, y, nodesize = 20, ntree=50)
RF_predict <- predict(RF_est, Xtest)

rf_loess_10 <- loess(RF_predict ~ c(1:length(RF_predict)), span=0.1)
rf_smoothed10 <- predict(rf_loess_10)


xval <- c(10*sin(pi * Xtest), rf_smoothed10.iid, rf_smoothed10)
xval_tag <- c(rep("Truth", length(10*sin(pi * Xtest))), rep("iid", length(rf_smoothed10.iid)),
              rep("correlated",length(rf_smoothed10)))

plot_data <- as.data.frame(xval)
plot_data$Methods <- xval_tag
coval <- c(rep(seq(0,1, by = 1/10000), 3))
plot_data$Covariate <- coval

ggplot(plot_data, aes(x=Covariate, y=xval, color=Methods)) +
  geom_point() + labs( x = "x") + labs( y = "m(x)") +
  theme(legend.title=element_blank(),text = element_text(size = 20)) +
  scale_color_manual(values = c("#F8766D", "#00BA38", "#619CFF"))

ggplot(plot_data %>% dplyr::filter(!(Methods=="correlated")), 
  aes(x=Covariate, y=xval, color=Methods)) +
  geom_point() + labs( x = "x") + labs( y = "m(x)") +
  theme(legend.title=element_blank(),text = element_text(size = 20)) +
  scale_color_manual(values = c("#00BA38", "#619CFF"))

set.seed(1)
#RFGLS_est <- RFGLS_estimate_spatial(coords, y, x, ntree = 50, cov.model = "exponential",nthsize = 20)
RFGLS_est <- RFGLS_estimate_spatial(coords, y, x, ntree = 50, cov.model = "exponential",
                                   nthsize = 20, param_estimate=T)

RFGLS_pred <- RFGLS_predict(RFGLS_est, Xtest)

rfgls_loess_10 <- loess(RFGLS_pred$predicted ~ c(1:length(Xtest)), span=0.1)
rfgls_smoothed10 <- predict(rfgls_loess_10)
rfgls.plot.data=data.frame(xval=rfgls_smoothed10,Methods="RFGLS",Covariate=Xtest)  

all.plot_data=rbind(plot_data,rfgls.plot.data)

all_plot_data <- data.frame(lapply(all.plot_data %>%
  dplyr::filter(!(Methods=='iid')), function(x) {gsub("correlated", "RF", x)})) %>%
  mutate(xval=as.numeric(xval),Covariate=as.numeric(Covariate))

ggplot(all_plot_data,aes(x=Covariate, y=xval, color=Methods)) +
  geom_point() + labs( x = "x") + labs( y = "m(x)") +
  theme(legend.title=element_blank(),text = element_text(size = 20)) +
  scale_color_manual(values = c("#F8766D","maroon", "#619CFF"))

df=data.frame(sx=coords[,1],sy=coords[,2],x=x,y=y,mu=10*sin(pi * x),error=y -10*sin(pi * x))
myplot(df,"error",col.br2)


##### impact of rounding/binning #####
## RFGLS timing example ##

set.seed(1124)
n=1000
x=matrix(rnorm(n),ncol=1)
beta=5
sigma=2
phi=3
tau=1

v=0.5
a=1.2
p0 = 5

mu= c(v*x + a*x^2)
plot(x,mu,ylim=range(mu)*c(0.5,1.5))

s=cbind(runif(n),runif(n))
dmat=as.matrix(dist(s))
w=c(rmvnorm(1,rep(0,n),sigma^2*exp(-phi*dmat)))
#w=w-mean(w)
y=rnorm(n,mu+w,tau)


t0=Sys.time()
set.seed(1)
rfgls.model.round1 = RFGLS_estimate_spatial(s,y,round(x,1),param_estimate=T)
t1=Sys.time()
t1-t0 ## Time difference of 1.842882 mins

rfgls.pred.round1=RFGLS_predict(rfgls.model.round1,x)
plot(x,mu,ylim=range(mu)*c(0.5,1.5))
points(x,rfgls.pred.round1$predicted,col="blue")
length(unique(round(x,1)))


t0=Sys.time()
set.seed(1)
rfgls.model.qntl = RFGLS_estimate_spatial(s,y,unname(matrix(round_to_quantile(x),ncol=1)),param_estimate = T)
t1=Sys.time()
t1-t0 ## Time difference of 1.938121 mins

rfgls.pred.qntl=RFGLS_predict(rfgls.model.qntl,x)
plot(x,mu,ylim=range(mu)*c(0.5,1.5))
points(x,rfgls.pred.qntl$predicted,col="green")
length(unique(round_to_quantile(x)))

t0=Sys.time()
set.seed(1)
rfgls.model.round2 = RFGLS_estimate_spatial(s,y,round(x,2),param_estimate = T)
t1=Sys.time()
t1-t0 ## Time difference of 8.278219 mins

rfgls.pred.round2=RFGLS_predict(rfgls.model.round2,x)

plot(x,mu,ylim=range(mu)*c(0.5,1.5))
points(x,rfgls.pred.round2$predicted,col="cyan")
length(unique(round(x,2)))

t0=Sys.time()
set.seed(1)
rfgls.model = RFGLS_estimate_spatial(s,y,x,param_estimate = T)
t1=Sys.time()
t1-t0 ## Time difference of 35.35141 mins

rfgls.pred=RFGLS_predict(rfgls.model,x)

plot(x,mu,ylim=range(mu)*c(0.5,1.5))
points(x,rfgls.pred$predicted,col="red")
length(unique(x))


##### impact of parallelization #####

t0=Sys.time()
set.seed(1)
rfgls.model = RFGLS_estimate_spatial(s,y,x,h=10,param_estimate = T)
t1=Sys.time()
t1-t0

rfgls.pred=RFGLS_predict(rfgls.model,x)

plot(x,mu,ylim=range(mu)*c(0.5,1.5))
points(x,rfgls.pred$predicted,col="red") ## Time difference of 21.63055 mins

##### mean shift example #####

## denser sampling ##
set.seed(5)
n <- 500 ## mean shift example change sample size to 500
coords <- cbind(runif(n,0,1), runif(n,0,1))
set.seed(10)
x <- as.matrix(runif(n),n,1)
sigma.sq = 10
phi = 1
tau.sq = 0.1
D <- as.matrix(dist(coords))
R <- exp(-phi*D)
w  <- rmvn(1, rep(0,n), sigma.sq*R)

beta=10
mu.gen =function(x) beta*pmax(0.5,x)

y <- rnorm(n, mu.gen(x) + w, sqrt(tau.sq))
plot(x,y)

# w  <- rmvn(1, rep(0,n), sigma.sq*R)
# y <- rnorm(n, 10*sin(pi * x), sqrt(tau.sq+var(w)))

Xtest <- matrix(seq(0,1, by = 1/10000), 10001, 1)

set.seed(1)
RF_est <- randomForest(x, y)
RF_predict <- predict(RF_est, Xtest)

rf_loess_10 <- loess(RF_predict ~ c(1:length(RF_predict)), span=0.1)
rf_smoothed10 <- predict(rf_loess_10)

xval <- c(mu.gen(Xtest), rf_smoothed10)
xval_tag <- c(rep("Truth", length(10*sin(pi * Xtest))), 
              rep("RF",length(rf_smoothed10)))

plot_data <- as.data.frame(xval)
plot_data$Methods <- xval_tag
coval <- rep(Xtest,2)
plot_data$Covariate <- coval

ggplot(plot_data, aes(x=Covariate, y=xval, color=Methods)) +
  geom_point() + labs( x = "x") + labs( y = "m(x)") +
  theme(legend.title=element_blank(),text = element_text(size = 20)) +
  scale_color_manual(values = c("#F8766D",  "#619CFF"))

set.seed(1)
#RFGLS_est <- RFGLS_estimate_spatial(coords, y, x, ntree = 50, cov.model = "exponential",nthsize = 20)
# RFGLS_est <- RFGLS_estimate_spatial(coords, y, unname(matrix(round_to_quantile(x),ncol=1)), ntree = 50, cov.model = "exponential",
#                                     nthsize = 20)
RFGLS_est <- RFGLS_estimate_spatial(coords, y, round(x,1), ntree = 50, cov.model = "exponential",
  nthsize = 20,sigma.sq=sigma.sq,phi=phi,tau.sq=tau.sq)


RFGLS_pred <- RFGLS_predict(RFGLS_est, Xtest)

rfgls_loess_10 <- loess(RFGLS_pred$predicted ~ c(1:length(Xtest)), span=0.1)
rfgls_smoothed10 <- predict(rfgls_loess_10)
rfgls.plot.data=data.frame(xval=rfgls_smoothed10,Methods="RFGLS",Covariate=Xtest)  

all.plot_data=rbind(plot_data,rfgls.plot.data)

ggplot(all.plot_data, aes(x=Covariate, y=xval, color=Methods)) +
  geom_point() + labs( x = "x") + labs( y = "m(x)") +
  theme(legend.title=element_blank(),text = element_text(size = 20)) +
  scale_color_manual(values = c("#F8766D","maroon", "#619CFF"))


## centered estimates ##

xval <- c(mu.gen(Xtest) -mean(mu.gen(Xtest)), rf_smoothed10 - mean(rf_smoothed10))
xval_tag <- c(rep("Truth", length(10*sin(pi * Xtest))), 
              rep("RF",length(rf_smoothed10)))

plot_data <- as.data.frame(xval)
plot_data$Methods <- xval_tag
coval <- rep(Xtest,2)
plot_data$Covariate <- coval

rfgls.plot.data=data.frame(xval=rfgls_smoothed10 -mean(rfgls_smoothed10),Methods="RFGLS",Covariate=Xtest)  

all.plot_data=rbind(plot_data,rfgls.plot.data)

ggplot(all.plot_data, aes(x=Covariate, y=xval, color=Methods)) +
  geom_point() + labs( x = "x") + labs( y = "m(x)") +
  theme(legend.title=element_blank(),text = element_text(size = 20)) +
  scale_color_manual(values = c("#F8766D","maroon", "#619CFF"))


#### same example but sampled from a larger domain
set.seed(5)
n <- 500 ## mean shift example change sample size to 500
coords <- cbind(runif(n,0,5), runif(n,0,5))
set.seed(10)
x <- as.matrix(runif(n),n,1)
sigma.sq = 10
phi = 1
tau.sq = 0.1
D <- as.matrix(dist(coords))
R <- exp(-phi*D)
w  <- rmvn(1, rep(0,n), sigma.sq*R)

beta=10
mu.gen =function(x) beta*pmax(0.5,x)

y <- rnorm(n, mu.gen(x) + w, sqrt(tau.sq))
plot(x,y)

# w  <- rmvn(1, rep(0,n), sigma.sq*R)
# y <- rnorm(n, 10*sin(pi * x), sqrt(tau.sq+var(w)))

Xtest <- matrix(seq(0,1, by = 1/10000), 10001, 1)

set.seed(1)
RF_est <- randomForest(x, y)
RF_predict <- predict(RF_est, Xtest)

rf_loess_10 <- loess(RF_predict ~ c(1:length(RF_predict)), span=0.1)
rf_smoothed10 <- predict(rf_loess_10)

xval <- c(mu.gen(Xtest), rf_smoothed10)
xval_tag <- c(rep("Truth", length(10*sin(pi * Xtest))), 
              rep("RF",length(rf_smoothed10)))

plot_data <- as.data.frame(xval)
plot_data$Methods <- xval_tag
coval <- rep(Xtest,2)
plot_data$Covariate <- coval

ggplot(plot_data, aes(x=Covariate, y=xval, color=Methods)) +
  geom_point() + labs( x = "x") + labs( y = "m(x)") +
  theme(legend.title=element_blank(),text = element_text(size = 20)) +
  scale_color_manual(values = c("#F8766D",  "#619CFF"))

set.seed(1)
#RFGLS_est <- RFGLS_estimate_spatial(coords, y, x, ntree = 50, cov.model = "exponential",nthsize = 20)
# RFGLS_est <- RFGLS_estimate_spatial(coords, y, unname(matrix(round_to_quantile(x),ncol=1)), ntree = 50, cov.model = "exponential",
#                                     nthsize = 20)
RFGLS_est <- RFGLS_estimate_spatial(coords, y, round(x,1), ntree = 50, cov.model = "exponential",
  nthsize = 20,sigma.sq=sigma.sq,phi=phi,tau.sq=tau.sq)


RFGLS_pred <- RFGLS_predict(RFGLS_est, Xtest)

rfgls_loess_10 <- loess(RFGLS_pred$predicted ~ c(1:length(Xtest)), span=0.1)
rfgls_smoothed10 <- predict(rfgls_loess_10)
rfgls.plot.data=data.frame(xval=rfgls_smoothed10,Methods="RFGLS",Covariate=Xtest)  

all.plot_data=rbind(plot_data,rfgls.plot.data)

ggplot(all.plot_data, aes(x=Covariate, y=xval, color=Methods)) +
  geom_point() + labs( x = "x") + labs( y = "m(x)") +
  theme(legend.title=element_blank(),text = element_text(size = 20)) +
  scale_color_manual(values = c("#F8766D","maroon", "#619CFF"))


##### impact of estimating spatial parameter #####
## using the same previous example ##
set.seed(5)
n <- 500 
coords <- cbind(runif(n,0,1), runif(n,0,1))
set.seed(10)
x <- as.matrix(runif(n),n,1)
sigma.sq = 10
phi = 2
tau.sq = 0.1
D <- as.matrix(dist(coords))
R <- exp(-phi*D)
w  <- rmvn(1, rep(0,n), sigma.sq*R)

beta=10
mu.gen =function(x) beta*pmax(0.5,x)

y <- rnorm(n, mu.gen(x) + w, sqrt(tau.sq))
plot(x,y)

# w  <- rmvn(1, rep(0,n), sigma.sq*R)
# y <- rnorm(n, 10*sin(pi * x), sqrt(tau.sq+var(w)))

Xtest <- matrix(seq(0,1, by = 1/10000), 10001, 1)

set.seed(1)
RF_est <- randomForest(x, y)
RF_predict <- predict(RF_est, Xtest)

rf_loess_10 <- loess(RF_predict ~ c(1:length(RF_predict)), span=0.2)
rf_smoothed10 <- predict(rf_loess_10)

xval <- c(mu.gen(Xtest), rf_smoothed10)
xval_tag <- c(rep("Truth", length(10*sin(pi * Xtest))), 
              rep("RF",length(rf_smoothed10)))

plot_data <- as.data.frame(xval)
plot_data$Methods <- xval_tag
coval <- rep(Xtest,2)
plot_data$Covariate <- coval

ggplot(plot_data, aes(x=Covariate, y=xval, color=Methods)) +
  geom_point() + labs( x = "x") + labs( y = "m(x)") +
  theme(legend.title=element_blank(),text = element_text(size = 20)) +
  scale_color_manual(values = c("#F8766D",  "#619CFF"))

set.seed(1)
#RFGLS_est <- RFGLS_estimate_spatial(coords, y, x, ntree = 50, cov.model = "exponential",nthsize = 20)
# RFGLS_est <- RFGLS_estimate_spatial(coords, y, unname(matrix(round_to_quantile(x),ncol=1)), ntree = 50, cov.model = "exponential",
#                                     nthsize = 20)
RFGLS_est <- RFGLS_estimate_spatial(coords, y, round(x,1), ntree = 50, cov.model = "exponential",
                                    nthsize = 20,param_estimate=T)

RFGLS_pred <- RFGLS_predict(RFGLS_est, Xtest)

rfgls_loess_10 <- loess(RFGLS_pred$predicted ~ c(1:length(Xtest)), span=0.2)
rfgls_smoothed10 <- predict(rfgls_loess_10)
rfgls.plot.data=data.frame(xval=rfgls_smoothed10,Methods="RFGLS: estimated pars",Covariate=Xtest)  


set.seed(1)
#RFGLS_est <- RFGLS_estimate_spatial(coords, y, x, ntree = 50, cov.model = "exponential",nthsize = 20)
# RFGLS_est <- RFGLS_estimate_spatial(coords, y, unname(matrix(round_to_quantile(x),ncol=1)), ntree = 50, cov.model = "exponential",
#                                     nthsize = 20)
RFGLS_est.true <- RFGLS_estimate_spatial(coords, y, round(x,1), ntree = 50, cov.model = "exponential",
  nthsize = 20,sigma.sq=sigma.sq,phi=phi,tau.sq=tau.sq)

RFGLS_pred.true <- RFGLS_predict(RFGLS_est.true, Xtest)

rfgls_loess_10.true <- loess(RFGLS_pred.true$predicted ~ c(1:length(Xtest)), span=0.2)
rfgls_smoothed10.true <- predict(rfgls_loess_10.true)
rfgls.plot.data.true=data.frame(xval=rfgls_smoothed10.true,Methods="RFGLS: true pars",Covariate=Xtest)  


set.seed(1)
#RFGLS_est <- RFGLS_estimate_spatial(coords, y, x, ntree = 50, cov.model = "exponential",nthsize = 20)
# RFGLS_est <- RFGLS_estimate_spatial(coords, y, unname(matrix(round_to_quantile(x),ncol=1)), ntree = 50, cov.model = "exponential",
#                                     nthsize = 20)
RFGLS_est.largephi <- RFGLS_estimate_spatial(coords, y, round(x,1), ntree = 50, cov.model = "exponential",
  nthsize = 20,phi=200,sigma.sq=sigma.sq,tau.sq=tau.sq)

RFGLS_pred.largephi <- RFGLS_predict(RFGLS_est.largephi, Xtest)

rfgls_loess_10.largephi <- loess(RFGLS_pred.largephi$predicted ~ c(1:length(Xtest)), span=0.2)
rfgls_smoothed10.largephi <- predict(rfgls_loess_10.largephi)
rfgls.plot.data.largephi=data.frame(xval=rfgls_smoothed10.largephi,Methods="RFGLS: large phi",Covariate=Xtest)  


all.plot_data=rbind(plot_data,rfgls.plot.data,rfgls.plot.data.true,rfgls.plot.data.largephi)

ggplot(all.plot_data, aes(x=Covariate, y=xval, color=Methods)) +
  geom_point() + labs( x = "x") + labs( y = "m(x)") +
  theme(legend.title=element_blank(),text = element_text(size = 20)) +
  scale_color_manual(values = c("#F8766D","maroon","green", "cyan","#619CFF"))


#################### Real data analysis ###################
### Plant richness data ###

pl.df=plant_richness_df %>% mutate(temp=normalize(climate_bio1_average)) 

## this plot make sit clear why RF-loc does the best, clear gradient with latitude ##
ggplot(pl.df,aes(x=x,y=y,col=log(richness_species_vascular))) +
  geom_point(size=2.5) + scale_color_gradientn(colors=c("midnightblue", "cyan", "yellow", "red"),name="log(richness)")+
  theme(text = element_text(size = 14))

ggplot(pl.df,aes(x=x,y=y,col=climate_bio1_average)) +
  geom_point(size=2.5) + scale_color_gradientn(colors=c("midnightblue", "cyan", "yellow", "red"),name="temperature")


plot(pl.df$climate_bio1_average,pl.df$richness_species_vascular,xlab="temp",ylab="richness")
plot(pl.df$climate_bio1_average,log(pl.df$richness_species_vascular),xlab="temp",ylab="log(richness)")

max.dist <- 0.75*max(rdist(unname(as.matrix(pl.df[,c("x","y")]))))
bins <- 20

vario1raw <- variog(coords=unname(as.matrix(pl.df[,c("x","y")])), data=pl.df$richness_species_vascular, 
                    uvec=(seq(0, max.dist, length=bins)))
plot(vario1raw,pch=16)



pl.df=na.omit(pl.df) #%>% dplyr::filter(richness_species_vascular < 15000)
pl.df=pl.df %>% mutate(richness_species_vascular=log(richness_species_vascular))
distance_matrix = as.matrix(dist(pl.df[,c("x","y")]))

### test train split

set.seed(1)
high.temp=which(pl.df$climate_bio1_average>200)
index=sample(high.temp,45)

pl.df.in=pl.df[-index,]
pl.df.out=pl.df[index,]

pl.df$data="train"
pl.df$data[index]="test"

ggplot(pl.df,aes(x=x,y=y,col=data)) +
  geom_point(size=2.5) +
  theme(text = element_text(size = 14))

plot(pl.df.in$climate_bio1_average,log(pl.df.in$richness_species_vascular),
     xlab="temp",ylab="log(richness)",col=hue_pal()(2)[2])
points(pl.df.out$climate_bio1_average,log(pl.df.out$richness_species_vascular),
     xlab="temp",ylab="log(richness)",col=hue_pal()(2)[1])
legend("topleft",c("test","train"),pch=1,col=hue_pal()(2))

## linear model ##

lm.est=lm(richness_species_vascular ~ temp, data=pl.df.in)
lm_pred=predict(lm.est,newdata=pl.df)

vario1raw <- variog(coords=unname(as.matrix(pl.df[,c("x","y")])), data=pl.df$richness_species_vascular-lm_pred,
                    uvec=(seq(0, max.dist, length=bins)))
plot(vario1raw,pch=16)
# 
rmse(pl.df.out$richness_species_vascular,lm_pred[index])

plot(pl.df$climate_bio1_average,pl.df$richness_species_vascular,xlab="temperature",ylab="log(richness)")
lines(sort(pl.df$climate_bio1_average),lm_pred[order(pl.df$climate_bio1_average)],col="red",lwd=2)


## BRISC ##
br <- BRISC_estimation(unname(as.matrix(pl.df.in[,c("x","y")])),
                       pl.df.in$richness_species_vascular, 
                       cbind(1,matrix(pl.df.in$temp,ncol=1)),n.neighbors = 15)

br.pred <- BRISC_prediction(br,unname(as.matrix(pl.df[,c("x","y")])),cbind(1,matrix(pl.df$temp,ncol=1)))

rmse(pl.df.out$richness_species_vascular,br.pred$prediction[index])
# 
br.mean.pred <- BRISC_prediction(br,100*unname(as.matrix(pl.df[,c("x","y")])),cbind(1,matrix(pl.df$temp,ncol=1)))
rmse(pl.df.out$richness_species_vascular,br.mean.pred$prediction[index])
# 
plot(pl.df$climate_bio1_average,pl.df$richness_species_vascular,xlab="temperature",ylab="log(richness)")
lines(sort(pl.df$climate_bio1_average),lm_pred[order(pl.df$climate_bio1_average)],col="red",lwd=2)
lines(sort(pl.df$climate_bio1_average),br.mean.pred$prediction[order(pl.df$climate_bio1_average)],col="blue",lwd=2)


## BRISC with latitude ##
br2 <- BRISC_estimation(unname(as.matrix(pl.df.in[,c("x","y")])),
                        pl.df.in$richness_species_vascular, 
                        cbind(1,unname(as.matrix(pl.df.in[,c("temp","y")]))),n.neighbors = 15)

br2.pred <- BRISC_prediction(br2,unname(as.matrix(pl.df[,c("x","y")])),cbind(1,unname(as.matrix(pl.df[,c("temp","y")]))))

rmse(pl.df.out$richness_species_vascular,br2.pred$prediction[index])

## RF ##

set.seed(1)
RF_est <- randomForest(richness_species_vascular ~ temp, data=pl.df.in)
RF_predict <- predict(RF_est, pl.df)

rmse(pl.df.out$richness_species_vascular,RF_predict[index])

## RFGLS ##

set.seed(1)

#pl.df$x=0.01*pl.df$x

RFGLS_est <- RFGLS_estimate_spatial(unname(as.matrix(pl.df.in[,c("x","y"),drop=F])),
  pl.df.in$richness_species_vascular, 
  matrix(pl.df.in$temp,ncol=1), ntree = 500, nthsize = 20,param_estimate = T)
#,sigma.sq=br$Theta[1],tau.sq=br$Theta[2],phi=br$Theta[3])

RFGLS_mean_pred <- RFGLS_predict(RFGLS_est,matrix(pl.df$temp,ncol=1))

RFGLS_sp_pred <- RFGLS_predict_spatial(RFGLS_est,
  unname(as.matrix(pl.df[,c("x","y"),drop=F])),matrix(pl.df$temp,ncol=1))

rmse(pl.df.out$richness_species_vascular,RFGLS_sp_pred$prediction[index])
rmse(pl.df.out$richness_species_vascular,RFGLS_mean_pred$predicted[index])
# 
rfgls_loess_10 <- loess(RFGLS_mean_pred$predicted ~ pl.df$temp, span=0.5)
# 
plot(pl.df$climate_bio1_average,pl.df$richness_species_vascular,xlab="temperature",ylab="log(richness)")
lines(sort(pl.df$climate_bio1_average),rfgls_loess_10$fitted[order(pl.df$climate_bio1_average)],col="red",lwd=2)

### predictions ###
## RFGLS with latitude as extra coordinates ##

set.seed(1)
RFGLS2_est <- RFGLS_estimate_spatial(unname(as.matrix(pl.df.in[,c("x","y")])),
  pl.df.in$richness_species_vascular, 
  unname(as.matrix(pl.df.in[,c("temp","y")])), ntree = 200, nthsize = 20,param_estimate = T)

RFGLS2_mean_pred <- RFGLS_predict(RFGLS2_est,unname(as.matrix(pl.df[,c("temp","y")])))

RFGLS2_sp_pred <- RFGLS_predict_spatial(RFGLS2_est,
  unname(as.matrix(pl.df[,c("x","y")])),unname(as.matrix(pl.df[,c("temp","y")])))

rmse(pl.df.out$richness_species_vascular,RFGLS2_sp_pred$prediction[index])
# rmse(pl.df.out$richness_species_vascular,RFGLS2_mean_pred$predicted[index])

## RF-loc ##

set.seed(1)
RF_loc_est <- randomForest(richness_species_vascular ~ temp + y + x, data=pl.df.in)
RF_loc_predict <- predict(RF_loc_est, pl.df)

rmse(pl.df.out$richness_species_vascular,RF_loc_predict[index])

## spRF ##
set.seed(1)
X=cbind(unname(matrix(pl.df$temp,ncol=1)),distance_matrix[,-index])
dimnames(X)=NULL
sprf_est=randomForest(X[-index,],pl.df.in$richness_species_vascular)
spRF_predict <- predict(sprf_est, newdata=X)

rmse(pl.df.out$richness_species_vascular,spRF_predict[index])

## Mean prediction error ##
df1=data.frame(fold=1,Metric="mean",Method=c("LM","spLM","RF","RFGLS"),
           RMSE=round(c(rmse(pl.df.out$richness_species_vascular,lm_pred[index]),
                        rmse(pl.df.out$richness_species_vascular,br.mean.pred$prediction[index]),
                        rmse(pl.df.out$richness_species_vascular,RF_predict[index]),
                        rmse(pl.df.out$richness_species_vascular,RFGLS_mean_pred$predicted[index])),2))


## Spatial prediction error ##
df2=data.frame(fold=1,Metric="spatial",Method=c("LM","spLM","spLM2","RF","RFGLS","RFGLS2","RFloc","spRF"),
           RMSE=round(c(rmse(pl.df.out$richness_species_vascular,lm_pred[index]),
                  rmse(pl.df.out$richness_species_vascular,br.pred$prediction[index]),
                  rmse(pl.df.out$richness_species_vascular,br2.pred$prediction[index]),
                  rmse(pl.df.out$richness_species_vascular,RF_predict[index]),
                  rmse(pl.df.out$richness_species_vascular,RFGLS_sp_pred$prediction[index]),
                  rmse(pl.df.out$richness_species_vascular,RFGLS2_sp_pred$prediction[index]),
                  rmse(pl.df.out$richness_species_vascular,RF_loc_predict[index]),
                  rmse(pl.df.out$richness_species_vascular,spRF_predict[index])),2))


rbind(df1,df2)

##### time-series example #####

df=read.csv("../data/colocation_hourly.csv",header = T) %>%
  mutate(Time=as.POSIXct(LST))

df.long = df %>% 
  select(Time,PM25_raw,PM25_MDE) %>%
  pivot_longer(-Time,names_to = 'Series',values_to="value") 

ggplot(df.long,aes(x=Time,y=value,col=Series),linewidth=1) +
  geom_path() +
  scale_color_manual(values = c("#E76BF3","#619CFF")) +
  ylab('PM25 in mu g m^-3')

ggplot(df,aes(x=PM25_MDE,y=PM25_raw),linewidth=1) +
  geom_point()

df.in=df[1:288,]
df.out=df[337:384,]

df.all=rbind(df.in,df.out)

l=lm(PM25_raw ~ PM25_MDE, data=df.in)
df.all$lmfit=predict(l,newdata=df.all)

p=pacf(df.all$PM25_raw-df.all$lmfit)

r=randomForest(PM25_raw ~ PM25_MDE, data=df.in)
df.all$rffit=predict(r,newdata=df.all)

y=df.in$PM25_raw
x=1*unname(as.matrix(df.in[,c("PM25_MDE"),drop=F],ncol=1))
X=1*unname(as.matrix(df.all[,c("PM25_MDE"),drop=F],ncol=1))
set.seed(1)
RFGLS <- RFGLS_estimate_timeseries(y, x, ntree = 500, param_estimate = T, lag_params = p$acf[1],
  nthsize = 20)

RFGLS_predict_temp_unknown <- RFGLS_predict(RFGLS,X)

df.all$rfglsfit=RFGLS_predict_temp_unknown$predicted

rmse(df.all$lmfit[-(1:288)],df.all$PM25_raw[-(1:288)])
rmse(df.all$rffit[-(1:288)],df.all$PM25_raw[-(1:288)])
rmse(df.all$rfglsfit[-(1:288)],df.all$PM25_raw[-(1:288)])

df.all=df.all %>% 
  mutate(set=c(rep('train',288),rep('test',48)),
         PM25_raw_fit_LM=lmfit,PM25_raw_fit_RF=rffit,PM25_raw_fit_RFGLS=rfglsfit) %>% 
  select(Time,set,PM25_raw,PM25_MDE,PM25_raw_fit_LM,PM25_raw_fit_RF,PM25_raw_fit_RFGLS) 

df.long=df.all %>%
  pivot_longer(-c(Time,set),names_to="Series")

ggplot(data=df.long %>% dplyr::filter(set=="train"),aes(x=Time,y=value,col=Series)) +
  facet_grid(. ~ set,scales = "free_x") +
  geom_path() +
  theme(axis.text.x=element_text(angle=45,vjust = .5)) +
  xlab('Time') +
  ylab('PM25 in mu g m^-3') +
  scale_color_manual(values = c("#E76BF3","#619CFF", "#00BF7D","#F8766D","maroon"))

ggplot(data=df.long %>% dplyr::filter(set=="test"),aes(x=Time,y=value,col=Series)) +
  facet_grid(. ~ set,scales = "free_x") +
  geom_path() +
  theme(axis.text.x=element_text(angle=45,vjust = .5)) +
  xlab('Time') +
  ylab('PM25 in mu g m^-3') +
  scale_color_manual(values = c("#E76BF3","#619CFF", "#00BF7D","#F8766D","maroon"))


## trying AR(2) model
set.seed(1)
RFGLS2 <- RFGLS_estimate_timeseries(y, x, ntree = 500, param_estimate = T, lag_params = c(p$acf[1],p$acf[2]),
                                   nthsize = 20)

RFGLS2_predict_temp_unknown <- RFGLS_predict(RFGLS2,X)

df.all$PM25_raw_fit_RFGLS2=RFGLS2_predict_temp_unknown$predicted


plot(df.all$PM25_raw_fit_RFGLS,df.all$PM25_raw_fit_RFGLS2,xlab="RFGLS AR(1) fit",ylab="RFGLS AR(2) fit")
abline(a=0,b=1,col="red")

## fitting AR(1) timeseries data using spatial RFGLS
  ## AR(1) time-series covariance == exponential GP covariance in 1 dimension

## creating a normalized time variable, to be used as space
df$normtime=(1:nrow(df))/24

df.in=df[1:288,]
df.out=df[337:384,]

df.all=rbind(df.in,df.out)


y=df.in$PM25_raw
x=1*unname(as.matrix(df.in[,c("PM25_MDE"),drop=F],ncol=1))
coords=cbind(1,unname(as.matrix(df.in[,c("normtime"),drop=F]))) ## making timepoints into 2-dimensional spatial coordinates
coords.all=1*unname(as.matrix(df.all[,c("normtime"),drop=F],ncol=1))
X=1*unname(as.matrix(df.all[,c("PM25_MDE"),drop=F],ncol=1))
set.seed(1)
RFGLS.sp <- RFGLS_estimate_spatial(coords, y, x, ntree = 500, param_estimate = T, 
  cov.model = "exponential", nthsize = 20)

RFGLS_predict_sp <- RFGLS_predict(RFGLS.sp,X)

df.all$PM25_raw_fit_RFGLS.sp=RFGLS_predict_sp$predicted

plot(df.all$PM25_raw_fit_RFGLS,df.all$PM25_raw_fit_RFGLS.sp,xlab="RFGLS AR(1) fit",ylab="RFGLS spatial fit")
abline(a=0,b=1,col="red")
