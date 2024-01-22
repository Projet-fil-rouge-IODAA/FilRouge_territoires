library(raster)

#unzip('C:/Users/matth/Downloads/Crops.zip',exdir='C:/Users/matth/Downloads/Crops')
setwd('C:/Users/matth/Downloads/Crops/Crops')

f <- list.files(pattern = "tif$")
r <- lapply(f,raster)
d <- lapply(r,dim)
t <- as.data.frame(do.call(rbind,d))
print(summary(t))

s <- lapply(r,getValues) %>% 
  lapply(summary)
st <- as.data.frame(do.call(rbind,s))
print(summary(st)) 


stsetwd('C:/Users/matth/Downloads/Crops/Crops/cloudy_images')

fc <- list.files(pattern = "tif$")
rc <- lapply(fc,raster)
dc <- lapply(rc,dim)
tc <- as.data.frame(do.call(rbind,dc))
print(summary(tc))

sc <- lapply(rc,getValues) %>% 
  lapply(summary)
stc <- as.data.frame(do.call(rbind,sc))
print(summary(stc))
