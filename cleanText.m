function cleanedText = cleanText(text)
    cleanedText = erasePunctuation(text);
    cleanedText = lower(cleanedText);
    cleanedText = regexprep(cleanedText, '\<a[^>]*>', '');
    cleanedText = regexprep(cleanedText, 'https?://\S+', '');
    cleanedText = regexprep(cleanedText, '\s+', ' ');
end
