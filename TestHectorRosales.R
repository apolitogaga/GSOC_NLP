## 
## By Héctor Apolo Rosales Pulido
## March 2015
## For the Google Summer of Code
## <<R Project for Statistical Computing>>>
## NLP project.
##

##
## Test
##
## Assume that you have a text (try any text) by using tm package load the text 
## for processing after that use (Bag of word model) to represent this text
## 

##
## Potential tasks
## 
# 1- Create a web interface to enter the questions.
# 2- Use (tm package)to read the question.
# 3- Use (NLP package) to convert the question to tokenizers then the matrix finally train
# model and predict score.
# 4- Use UIMA to discover Knowledge.
# 5- Create database that will save answers for the questions.

##
## Skills required 
##
# a very good programming experience in R language and Java.
# Good programming Knowledge in HTML.
# Database and SQL.
# Knowledge about tm package and NLP package.


#
# I'm going to work with the reuters dataset.
# unlike the orignal data set this was a little bit modified, 
#      all articles with no topics were dropped, 
#      and we only have the first topic for each article.
# 

library(wordcloud)  # for word clouds
library(tm) # for topic modelling
library(SnowballC) # needed for stemming
library(e1071) # naiveBayes

reuters <- read.table("reuters.txt.gz", header=TRUE)
dim(reuters) # we have 2 columns and 9520 rows
(N <- dim(reuters)[1]) # save the size of our data

reuters$Content <- as.character(reuters$Content) # R loaded it as a factor, we change it to character.
#we get the number of topics.
(l <- length(levels(reuters$Topic)))
#how many topics have each one of them.
(tops <- table(reuters$Topic))

# these are the topcics moer frequent than the average frequency
(selected.tops <- tops[tops>N/l])

## Let's work only with these, for simplicity:
reuters.freq <- reuters[reuters$Topic %in% names(selected.tops),]

## The resulting data frame contains 7,873 news items on 9 topics. The actual news text is the column 
## "Content" and its category is the column "Topic". Possible goals are visualizing the data and creating a classifier
## for the news articles (we'll we doing both several times during the course)

(N.freq <- dim(reuters.freq)[1])  # new number of rows

levels(reuters.freq$Topic)
# re-level the factor to have only 9 levels (VERY IMPORTANT)
reuters.freq$Topic <- factor(reuters.freq$Topic)
levels(reuters.freq$Topic)

## an example of a text about 'money-fx'
reuters.freq[130,]
## an example of a text about 'sugar'
reuters.freq[134,]

## some entries are quite long ...
reuters.freq[133,]
## We first transform the data to a Corpus to have access to the nice {tm} routines for text manipulation
reuters.cp <- VCorpus (VectorSource(as.vector(reuters.freq$Content)) )
reuters.cp[5]$content
inspect(reuters.cp[1:2])

## so we pre-process the Corpus a bit
reuters.cp <- tm_map(reuters.cp, stripWhitespace) # Elimination of extra whitespaces
reuters.cp <- tm_map(reuters.cp, removeNumbers) # Elimination of numbers
reuters.cp <- tm_map(reuters.cp, removePunctuation) # Elimination of punctuation marks
reuters.cp <- tm_map(reuters.cp, content_transformer(tolower)) # Conversion to lowcase 

# Removal of English generic and custom stopwords
my.stopwords <- c(stopwords('english'), 'reuter', 'reuters', 'said')
reuters.cp <- tm_map(reuters.cp, removeWords, my.stopwords)
?tm_map # interface to apply transformation cuntions to corpora.

# Stemming (reducing inflected or derived words to their word stem, base or root form)
reuters.cp <- tm_map(reuters.cp, stemDocument, language="english", lazy=TRUE)

## Convert to TermDocumentMatrix
tdm <- TermDocumentMatrix (reuters.cp)
inspect(tdm[80:100,20:50]) ## 'inspect' displays detailed information on a corpus or a term-document matrix:
# So we really have a very sparse data representation
# inspect most popular words
findFreqTerms(tdm, lowfreq=1000)

## now we can form frequency counts
v <- sort(rowSums(as.matrix(tdm)),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v/sum(v))
# do
wordcloud(d$word,d$freq,scale=c(8,.5),max.words=100, random.order=FALSE)

## Now let's do something better; we generate a new term-document matrix by 
## TfIdf (weight by term frequency - inverse document frequency)
word.control <- list(weighting = function(x) weightTfIdf(x, normalize = TRUE))
tdm2 <- TermDocumentMatrix (reuters.cp, control=word.control)

v <- sort(rowSums(as.matrix(tdm2)),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v/sum(v))

## which possibly gives a better impression of the data, but this is subjective
wordcloud(d$word,d$freq,scale=c(8,1),max.words=50, random.order=FALSE)


## Let's work only with a couple of topics, for simplicity:
selected.tops <- c('ship','sugar')
reuters.ss <- reuters[reuters$Topic %in% selected.tops,]
## The goal is to classify (and be able to predict) news documents as dealing with ships or with sugar
## The resulting data now contains 305 news items
(N.freq <- dim(reuters.ss)[1])  # new number of rows
reuters.ss$Topic <- factor(reuters.ss$Topic)  # re-level the factor to have only 2 levels
## an example of a text about 'ship'
reuters.ss[130,]
## an example of a text about 'sugar'
reuters.ss[134,]

## One of the most used ML representations for text is the "bag of words"
documents <- reuters.ss$Content

## first we need some auxiliary functions to actually create the structure
source('reuters-aux.R')

## "pre-allocate" an empty list of the required length
bagged <- vector("list", length(documents))

## Now generate bag of words as a list
bagged <- lapply(documents,strip.text) # produce the stripped text
bagged[[1]]
bagged.bow <- lapply(bagged,table)# produce the bag of words
bagged.bow[[1]]
reuters.BOWs <- make.BoW.frame (bagged.bow)# make it a dataframe zZ
reuters.BOWs[1,"year"]
reuters.BOWs[1,"when"]
dim(reuters.BOWs)
## we have 305 news entries "described" by nearly 2,800 features (the words)
# Now we weight by inverse document frequency
reuters.BOWs.tfidf <- tfidf.weight(reuters.BOWs)

# and normalize by vector length
reuters.BOWs.tfidf <- div.by.euc.length(reuters.BOWs.tfidf)
dim(reuters.BOWs.tfidf)

# let's inspect the result
summary(colSums(reuters.BOWs.tfidf))

## too many columns
# remove those words shorter than 3 characters
reuters.BOWs.tfidf <- subset.data.frame (reuters.BOWs.tfidf, 
                        select=sapply(colnames(reuters.BOWs.tfidf), FUN=nchar)>2)
dim(reuters.BOWs.tfidf)

# remove those words whose total sum is not greater than the third quartile of the distribution
(r.3rdQ <- summary(colSums(reuters.BOWs.tfidf))[5])
reuters.BOWs.tfidf <- subset.data.frame (reuters.BOWs.tfidf,
                                         select=colSums(reuters.BOWs.tfidf)>r.3rdQ)
# obviously this is roughly a further 75% reduction, corresponding to the less represented words
dim(reuters.BOWs.tfidf)

# Add class labels back (the "Topics")
# (the normalizing and weighting functions don't work well with non-
# numeric columns so it's simpler to add the labels at the end)
reuters.definitive <- data.frame(Topic=reuters.ss$Topic,reuters.BOWs.tfidf)
dim(reuters.definitive)
# a final look at the first 10 variables
summary(reuters.definitive[,1:10])

## Naive Bayes Classifier
set.seed (1234)
N <- nrow(reuters.definitive)                                                                                              
learn <- sample(1:N, round(0.67*N))
nlearn <- length(learn)
ntest <- N - nlearn
reuters.nb <- naiveBayes (Topic ~ ., data=reuters.definitive[learn,], laplace=3)
# predict the left-out data
pred <- predict(reuters.nb,newdata=reuters.definitive[-learn,])
(tt <- table(pred,reuters.definitive$Topic[-learn]))
(error <- 100*(1-sum(diag(tt))/sum(tt)))
# so error is 17.8%, although there is a marked tendency to predict 'sugar'. The majority class is:
(baseline <- 100*(1 - max(table(reuters.definitive$Topic))/nrow(reuters.definitive)))
# so all errors below 42% are an improvement over the baseline
100*(baseline-error)/baseline
# actually we are able to get a relative reduction of 57.5% in error
## However note that this result is highly unstable, 
##   given the small size of both the learn and test sets