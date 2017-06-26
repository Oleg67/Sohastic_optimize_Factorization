filterRuns <- function(buildData, testData) {
  # Filter build data to include only runnners/jockeys/trainers found in the testData
  testRunners <- unique(testData[, 'runner'])
  buildRunners <- buildData[, 'runner'] %in% testRunners
  testMask <- testData[, 'runner'] %in% buildData[buildRunners, 'runner']
  testGoing <- unique(testData[testMask, 'going'])
  buildGoing <- buildData[, 'going'] %in% testGoing
  testObstacle <- unique(testData[testMask, 'obstacle'])
  buildObstacle <- buildData[, 'obstacle'] %in% testObstacle
  testGroups <- unique(testData[testMask, 'groups'])
  buildGroups <- buildData[, 'groups'] %in% testGroups
  buildMask <- buildRunners & buildGoing & buildObstacle & buildGroups
  testMask <- testMask & testData[, 'runner'] %in% buildData[buildMask, 'runner']
  list(buildMask = buildMask, testMask = testMask)
}  

fullSeq <- function(from, to, by) {
  # Create a sequence that always includes the last element
  fullS <- seq(from = from, to = to, by = by)
  if(to != fullS[length(fullS)]) {
    # add the end point
    fullS <- c(fullS, to+1)
  } else {
    fullS[length(fullS)] <- to+1
  }
  fullS
}