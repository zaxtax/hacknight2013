# Load the Data
setwd("~/talks/hack_night2013/")
data <- read.csv("fb_attendees.csv", header=T)

# Read off summaries

summary(data)

# Read off counts
length(data$Category[data$Category == "Concert venue"])
table(data$Category)
plot(sort(table(data$Category), decreasing=TRUE))
head(sort(table(data$Category), decreasing=T),n=20)
plot(head(sort(table(data$Category), decreasing=TRUE),n=20))

# Removing infrequent categories
which(table(data$Category) < 500)
names(which(table(data$Category) < 500))
data$Category %in% names(which(table(data$Category) < 500))

levels(data$Category) <- c(levels(data$Category), "Other")
data$Category[data$Category %in% names(which(table(data$Category) < 500))] <- as.factor("Other")

# Save after cleaning
write.csv(data,"fb_attendees_cleaned.csv")

