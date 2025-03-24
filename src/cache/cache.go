package cache

import (
	"context"
	"encoding/json"
	goerrors "errors"
	"fmt"

	"github.com/pkg/errors"
	"github.com/redis/go-redis/v9"
	"purify/src/utils"
)

var ErrNoAnswer = errors.New("no answer")

const keyPrefix = "go"

type Cache struct {
	client *redis.Client
}

func NewCache(client *redis.Client) *Cache {
	return &Cache{client: client}
}

type Value struct {
	Preconception []string `json:"preconception"`
	Agitation     []string `json:"agitation"`
}

func (c *Cache) GetAnswer(ctx context.Context, text string) (Value, error) {
	key := utils.Hash(text)
	fmt.Printf("[DEBUG] GetAnswer key=%s\n", key)

	valueString, err := c.get(ctx, key)
	if err != nil {
		if goerrors.Is(err, redis.Nil) {
			return Value{}, ErrNoAnswer
		}
		return Value{}, errors.Wrap(err, "failed to get answer")
	}

	var value Value
	if err = json.Unmarshal([]byte(valueString), &value); err != nil {
		return Value{}, errors.Wrap(err, "failed to unmarshal answer")
	}

	return value, nil
}

func (c *Cache) SetAnswer(ctx context.Context, text string, answer Value) error {
	key := utils.Hash(text)
	fmt.Printf("[DEBUG] SetAnswer key=%s\n", key)

	valueBytes, err := json.Marshal(answer)
	if err != nil {
		return errors.Wrap(err, "failed to marshal value")
	}

	if err = c.set(ctx, key, string(valueBytes)); err != nil {
		return errors.Wrap(err, "failed to set value")
	}

	return nil
}

func (c *Cache) get(ctx context.Context, key string) (string, error) {
	return c.client.Get(ctx, getKey(key)).Result()
}

func (c *Cache) set(ctx context.Context, key string, value string) error {
	return c.client.Set(ctx, getKey(key), value, 0).Err()
}

func getKey(key string) string {
	return fmt.Sprintf("%s:%s", keyPrefix, key)
}
