source("RollingForecast.R")
source("Tools.R")
source("FitLME.R")

data <- read.csv('Data/data.csv', header = TRUE, stringsAsFactors = FALSE)
data[data$build == "False", 'build'] <- FALSE
data[data$build == "True", 'build'] <- TRUE
data[data$test == "False", 'test'] <- FALSE
data[data$test == "True", 'test'] <- TRUE

timeIndex <- as.Date(as.POSIXlt(data$start_time, origin = "1970-01-01", tz = 'UTC'))
data['dates'] <- timeIndex

build <- as.logical(data$build)
test <- as.logical(data$test)

# Fix groups
bd <- data[build, ]
bd$groups <- -1
bd[bd$speed < 12 & bd$obstacle == 'F', 'groups'] <- 0
bd[bd$obstacle != 'F', 'groups'] <- 1
bd[bd$obstacle != 'F' & bd$speed < 10, 'groups'] <- 0
bd[bd$speed >= 12 & bd$obstacle == 'F', 'groups'] <- 2

td <- data[test, ]
td$groups <- -1
td[td$speed < 12 & td$obstacle == 'F', 'groups'] <- 0
td[td$obstacle != 'F', 'groups'] <- 1
td[td$obstacle != 'F' & td$speed < 10, 'groups'] <- 0
td[td$speed >= 12 & td$obstacle == 'F', 'groups'] <- 2

data$groups <- -1
data[build, 'groups'] <- bd$groups
data[test, 'groups'] <- td$groups

# Run forecast
system.time (res <- rollingForecast(x = data, model = 'fitLME', target = 'speed', k = 10))

write.csv(data.frame(res, stringsAsFactors = FALSE), file = "roll_normspeed.csv", row.names = FALSE)
