package utils

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestSplitTextIntoChunks(t *testing.T) {
	text := "Вот пример текста, который нужно будет       разделить на кусочки по 30 слов. " +
		"Этот текст может быть \t\n очень большим и содержать множество пробелов, отступов и переносов строк. " +
		"Таким образом, важно сохранить    \n   \t\t  все эти элементы при разделении   и последующей склейке текста обратно."

	chunkSize := 30
	chunks := SplitTextIntoChunks(text, chunkSize)

	require.Equal(t, text, strings.Join(chunks, ""))
}
