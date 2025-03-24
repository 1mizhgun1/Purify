package utils

import (
	"fmt"
	"hash/fnv"
)

func Hash(s string) string {
	h := fnv.New64a()
	_, _ = h.Write([]byte(s))
	return fmt.Sprintf("%d", h.Sum64())
}
