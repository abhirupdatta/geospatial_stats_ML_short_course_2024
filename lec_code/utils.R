### Contains code snippets used by scripts for multiple lectures
## multivariate normal RNG ##
rmvn <- function(n, mu = 0, V = matrix(1)){
  p <- length(mu)
  if(any(is.na(match(dim(V),p))))
    stop("Dimension not right!")
  D <- chol(V)
  t(matrix(rnorm(n*p), ncol=p)%*%D + rep(mu,rep(n,p)))
}


### color palettes

col.br <- colorRampPalette(c("midnightblue", "cyan", "yellow", "red"))
col.pal <- col.br(5)

col.br2 <- colorRampPalette(c("midnightblue", "white", "red"))
col.pal2 <- col.br2(5)

### function for plotting interpolated surface of a column of a data table
myplot=function(tab,colname,pal=col.br){
  
  surf <- mba.surf(tab[,c("sx","sy",colname)], no.X=100, no.Y=100, h=5, m=2, extend=FALSE)$xyz.est
  image.plot(surf, xaxs = "r", yaxs = "r", xlab="Easting (m)", ylab="Northing (m)", col=pal(25))
  
}


round_to_quantile <- function(vec) {
  # Generate the quantiles at 2% increments
  quantiles <- quantile(vec, probs = seq(0, 1, by = 0.02))
  
  # Define a function to round each value to the nearest quantile
  round_to_nearest <- function(x) {
    closest_quantile <- quantiles[which.min(abs(quantiles - x))]
    return(closest_quantile)
  }
  rounded_vec <- sapply(vec, round_to_nearest)
  
  return(rounded_vec)
}


normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}


rmse = function(x,y){
  sqrt(mean((x-y)^2))
}
