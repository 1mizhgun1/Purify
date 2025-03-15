package utils

import (
	"regexp"
	"strings"
)

var re = regexp.MustCompile(`(\S+)(\s*)`)

func splitTextWithDelimiters(text string) [][]string {
	return re.FindAllStringSubmatch(text, -1)
}

func flattenSplitResult(splitResult [][]string) []string {
	var words []string
	for _, match := range splitResult {
		words = append(words, match[1]+match[2])
	}
	return words
}

func SplitTextIntoChunks(text string, chunkSize int) []string {
	splitResult := splitTextWithDelimiters(text)
	words := flattenSplitResult(splitResult)
	var chunks []string
	var chunk strings.Builder
	wordCount := 0

	for _, word := range words {
		chunk.WriteString(word)
		if len(strings.TrimSpace(word)) > 0 {
			wordCount++
		}
		if wordCount == chunkSize {
			chunks = append(chunks, chunk.String())
			chunk.Reset()
			wordCount = 0
		}
	}

	if chunk.Len() > 0 {
		chunks = append(chunks, chunk.String())
	}

	return chunks
}
