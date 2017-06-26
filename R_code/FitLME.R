library('lme4')

fitLME <- function(x) {
  lme <- lmer(speed ~ distance + dummy(groups) + dummy(obstacle) + dummy(going) + (1|runner), data = x)
  lme
}