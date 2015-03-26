####################################################################
# Functions for document retrieval and simple similarity searching
# Modified from code written by Tom Minka (thanks!)
####################################################################

# Rescale rows of an array or data frame by a given weight vector

scale.rows <- function(x,w) apply(x,2,function(x){x*w})

# Rescale columns of an array or data frame by a given weight vector

scale.cols <- function(x,w) t(apply(x,1,function(x){x*w}))


# Normalize vectors by the sum of their entries
# Input assumed to be a set of vectors in array form, one vector per row

div.by.sum <- function(x) scale.rows(x,1/(rowSums(x)+1e-16))

# Normalize vectors by their Euclidean length
# Input assumed to be a set of vectors in array form, one vector per row

div.by.euc.length <- function(x) scale.rows(x,1/sqrt(rowSums(x^2)+1e-16))


# Turn a string into a vector of words
# for comparability across bags of words, also strip out punctuation and
# numbers, and shift all letters into lower case

strip.text <- function(txt) 
{
  # remove apostrophes (so "don't" -> "dont", "Jane's" -> "Janes", etc.)
  txt <- gsub("'","",txt)
  # convert to lowercase
  txt <- tolower(txt)
  # change other non-alphanumeric characters to spaces
  txt <- gsub("[^a-z0-9]"," ",txt)
  # change digits to # and remove them
  txt <- gsub("[0-9]+","",txt)
  
  # split and make one vector
  txt <- unlist(strsplit(txt," "))
  # remove empty words
  txt <- txt[txt != ""]
  txt
}

# Remove columns from a ragged array which only appear in one row

remove.singletons.ragged <- function(x) 
{
  # Collect all the column names, WITH repetition
  col.names <- c()
  for(i in 1:length(x))
  {
    col.names <- c(col.names, names(x[[i]]))
  }
  # See how often each name appears
  count <- table(col.names)
  # Loop over vectors and keep only the columns which show up more than once
  for(i in 1:length(x)) 
  {
    not.single <- (count[names(x[[i]])] > 1)
    x[[i]] <- x[[i]][not.single]
  }
  x
}

# Standardize a ragged array so all vectors have the same length and ordering
# Supplies NAs for missing values
# Input: a list of vectors with named columns
# Output: a standardized list of vectors with named columns

standardize.ragged <- function(x) 
{
  # Keep track of all the column names from all the vectors in a single vector
  col.names <- c()
  # Get the union of column names by iterating over the vectors - using
  # setdiff() is faster than taking unique of the concatenation, the more
  # obvious approach
  for(i in 1:length(x)) 
  {
    col.names <- c(col.names, setdiff(names(x[[i]]),col.names))
  }
  # put the column names in alphabetical order, for greater comprehensibility
  col.names <- sort(col.names)
  # Now loop over the vectors again, putting them in order and filling them out
  # Note: x[[y]] returns NA if y is not the name of a column in x
  for (i in 1:length(x)) 
  {
    x[[i]] <- x[[i]][col.names]
    # Make sure the names are right
    names(x[[i]]) <- col.names
  }
  x
}


# Turn a list of bag-of-words vectors into a data frame, one row per bag
# Input: 
#   list of bag-of-words vectors
#   list of row names (optional),
#   flag for whether singletons should be removed,
#   flag for whether words missing in a document should be coded 0
# Output: data frame, columns named by the words and rows matching documents

make.BoW.frame <- function(x,row.names,remove.singletons=TRUE,
                           absent.is.zero=TRUE) 
{
  # Should we remove one-time-only words?
  if (remove.singletons) { y <- remove.singletons.ragged(x) }
  else { y <- x }
  
  # Standardize the column names
  y <- standardize.ragged(y)
  
  # Transform the list into an array (there are probably slicker ways to do this)
  z = y[[1]] # Start with the first row
  if (length(y) > 1) { # More than one row?
    for (i in 2:length(y)) 
    {
      z = rbind(z,y[[i]],deparse.level=0) # then stack them
    }
  }
  
  # Make the data frame and use row names, if provided
  if (missing(row.names)) { BoW.frame <- data.frame(z) }
  else { BoW.frame <- data.frame(z,row.names=row.names) }

  if (absent.is.zero) 
  {
    # The standardize.ragged function maps missing words to "NA", so replace
    # those with zeroes to simplify calculation
    BoW.frame <- apply(BoW.frame,2,function(q){ifelse(is.na(q),0,q)})
  }
  BoW.frame
}


# Compute inverse document frequency weights and rescale a data frame

tfidf.weight <- function(x) 
{
  # TFIDF weighting
  doc.freq <- colSums(x>0)
  doc.freq[doc.freq == 0] <- 1
  scale.cols(x, log(nrow(x)/doc.freq))
}


