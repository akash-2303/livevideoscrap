# Algorithm:
Credits: [Dr. Christian Grant](https://ceg.me/)

1. Vectorize (using embedding vector) each sentence/paragraph in the transcript. (You can use Spacy to get the vector). You should add vector and time for each sentence and paragraph over the comments.
2. In a sliding window, create the average vector for each moment of time. Create the sliding window size that makes sense to you. Find the average vector of each window. (This will give you the average signal in the window range.)
3. We'll assume that comments close to this average signal (using vector distance) will be linked to the closest, most recent transcript vector.
4. For each comment vector, check to see if which window it is closest to. If closer to an older window's average, then that comment should be linked to the older comment. You can use the vector distances as a weighted distance.
