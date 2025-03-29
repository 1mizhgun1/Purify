package utils

import (
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestSplitTextIntoChunks(t *testing.T) {
	text := "Вот пример текста, который нужно будет       разделить на кусочки по 30 слов. " +
		"Этот текст может быть \t\n очень большим и содержать множество пробелов, отступов и переносов строк. " +
		"Таким образом, важно сохранить    \n   \t\t  все эти элементы при разделении   и последующей склейке текста обратно."

	chunks := SplitTextIntoChunks(text, 30, 4)

	require.Equal(t, text, strings.Join(chunks, ""))
}

func TestSplitBlocksIntoChunks(t *testing.T) {
	blocks := []string{
		"Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
		"Vestibulum fermentum massa eu magna vestibulum, vel convallis tortor gravida.",
		"Curabitur dictum tincidunt purus, eget malesuada magna imperdiet at.",
		"A verylongwordthatislongerthanmaxTokensshouldbesplitproperly",
	}
	maxTokens := 500
	chunks := SplitBlocksIntoChunks(blocks, maxTokens)
	for i, chunk := range chunks {
		fmt.Printf("Chunk %d: %q\n", i+1, chunk)
	}
}

func TestFindSubstrings(t *testing.T) {
	block := "Welcome to Golang"

	substrings := []string{
		"Hello",
		"Go",
		"Golang",
	}

	result := FindSubstrings(block, substrings)

	for _, subs := range result {
		fmt.Printf("%v\n", subs)
	}
}
