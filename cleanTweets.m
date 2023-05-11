
function cleanedTweets = cleanTweets(tweets)
    cleanedTweets = tweets;
    for i = 1:numel(tweets)
        tweet = tweets{i};
        tweet = regexprep(tweet, 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ''); % Remove URLs
        tweet = regexprep(tweet, '@(?:[a-zA-Z]|[0-9]|[_])+', ''); % Remove mentions
        tweet = regexprep(tweet, '[^a-zA-Z\s]', ''); % Remove special characters and digits
        cleanedTweets{i} = tweet;
    end
end