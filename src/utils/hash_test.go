package utils

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestHash(t *testing.T) {
	str := "abc def 12345 hello мир"
	hash := Hash(str)

	for i := 0; i < 10; i++ {
		require.Equal(t, hash, Hash(str))
	}
}
