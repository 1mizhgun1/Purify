package utils

import (
	"regexp"
	"strings"
	"unicode"
)

var (
	re      = regexp.MustCompile(`(\S+)(\s*)`)
	countRe = regexp.MustCompile(`\s+`)
)

func countWords(text string) int {
	words := countRe.Split(text, -1)
	wordCount := 0
	for _, word := range words {
		if word != "" {
			wordCount++
		}
	}
	return wordCount
}

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

func SplitTextIntoChunks(text string, chunkSize int, maxChunks int) []string {
	if count := countWords(text); count > maxChunks*chunkSize {
		chunkSize = count / maxChunks
	}

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

func SplitBlocksIntoChunks(blocks []string, maxTokens int) []string {
	var chunks []string
	var currentChunk strings.Builder
	currentLength := 0

	addChunk := func() {
		if currentChunk.Len() > 0 {
			chunks = append(chunks, currentChunk.String())
			currentChunk.Reset()
		}
		currentLength = 0
	}

	for _, block := range blocks {
		if len(block) <= maxTokens {
			if currentLength+len(block) > maxTokens {
				addChunk()
			}
			if currentChunk.Len() > 0 {
				currentChunk.WriteString(" ")
			}
			currentChunk.WriteString(block)
			currentLength += len(block)
		} else {
			words := splitByWhitespace(block)
			var tempChunk strings.Builder
			tempLength := 0
			for _, word := range words {
				if tempLength+len(word) > maxTokens {
					chunks = append(chunks, tempChunk.String())
					tempChunk.Reset()
					tempLength = 0
				}
				if tempChunk.Len() > 0 {
					tempChunk.WriteString(" ")
				}
				tempChunk.WriteString(word)
				tempLength += len(word)
			}
			if tempChunk.Len() > 0 {
				chunks = append(chunks, tempChunk.String())
			}
		}
	}
	addChunk()

	return chunks
}

func splitByWhitespace(text string) []string {
	var words []string
	var word strings.Builder
	for _, r := range text {
		if unicode.IsSpace(r) || r == '\n' || r == '\t' {
			if word.Len() > 0 {
				words = append(words, word.String())
				word.Reset()
			}
		} else {
			word.WriteRune(r)
		}
	}
	if word.Len() > 0 {
		words = append(words, word.String())
	}
	return words
}

func FindSubstrings(block string, substrings []string) []string {
	result := make([]string, 0)

	for _, sub := range substrings {
		if strings.Contains(block, sub) {
			result = append(result, sub)
		}
	}

	return result
}
