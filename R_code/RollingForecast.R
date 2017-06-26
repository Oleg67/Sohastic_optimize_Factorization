

rollingForecast <- function(x, model, target, k) {
  # Divide data into build and test sets
  buildMask <- as.logical(x$build)
  testMask <- as.logical(x$test)
  buildData <- x[buildMask, ]
  testData <- x[testMask, ]
  testDays <- unique(x$dates[testMask])
  b_d <- buildData
  
  # Containers for prediction and residuals
  isFit <- matrix(rep(NA, nrow(b_d)), ncol = 1); isRes <- isFit 
  oosFit <- NULL; oosRes <- NULL
  
  breaks <- fullSeq(from=1, to=length(testDays), by=k)
  
  for (i in 1:(length(breaks)-1)) {
    start_idx <- breaks[i]; end_idx <- breaks[i+1]-1
    cat('Window ', i, ': ', as.character(testDays[start_idx]), ' - ', as.character(testDays[end_idx]), '\n')
    windowDays <- testDays[start_idx:end_idx] # days in a test window
    t_d <- testData[testData$dates %in% windowDays, ] # data in a test window
    filtMasks <- filterRuns(buildData = b_d, testData = t_d) # sync rand. eff. levels
    modObj <- get(model)(b_d[filtMasks[['buildMask']], ]) # fit model
    
    if (i == 1) { # In-sample fit
      isFit[filtMasks[['buildMask']], ] <- fitted(modObj)
      isRes[filtMasks[['buildMask']], ] <- b_d[filtMasks[['buildMask']], target] - fitted(modObj)
    }
    
    # Prediction and residuals for each test window
    pred <- matrix(rep(NA, nrow(t_d)), ncol=1)
    res <- pred
    pred[filtMasks[['testMask']], ] <- predict(modObj, t_d[filtMasks[['testMask']], ], allow.new.levels=TRUE)
    res[filtMasks[['testMask']], ] <- (t_d[filtMasks[['testMask']], target] - pred[filtMasks[['testMask']], ])
    oosFit <- rbind(oosFit, pred)
    oosRes <- rbind(oosRes, res)
    
    # Add last test window to build data
    b_d <- rbind(b_d, t_d)
  }
  
  
  prediction <- rep(NA, nrow(x))
  residuals <- prediction
  
  prediction[buildMask] <- isFit
  prediction[testMask] <- oosFit
  residuals[buildMask] <- isRes
  residuals[testMask] <- oosRes
  
  list(prediction = prediction, residuals = residuals)
}

