package cache

import (
	"context"
	goerrors "errors"
	"fmt"

	"github.com/pkg/errors"
	"github.com/redis/go-redis/v9"
	"purify/src/utils"
)

var ErrNoAnswer = errors.New("no answer")

type Cache struct {
	client *redis.Client
}

func NewCache(client *redis.Client) *Cache {
	return &Cache{client: client}
}

func (c *Cache) GetAnswer(ctx context.Context, text string, feature string) (string, error) {
	key := utils.Hash(text)

	value, err := c.get(ctx, key, feature)
	if err != nil {
		if goerrors.Is(err, redis.Nil) {
			return "", ErrNoAnswer
		}
		return "", errors.Wrap(err, "failed to get answer")
	}

	return value, nil
}

func (c *Cache) SetAnswer(ctx context.Context, text string, answer string, feature string) error {
	key := utils.Hash(text)

	if err := c.set(ctx, key, answer, feature); err != nil {
		return errors.Wrap(err, "failed to set value")
	}

	return nil
}

func (c *Cache) get(ctx context.Context, key string, keyPrefix string) (string, error) {
	return c.client.Get(ctx, getKey(key, keyPrefix)).Result()
}

func (c *Cache) set(ctx context.Context, key string, value string, keyPrefix string) error {
	return c.client.Set(ctx, getKey(key, keyPrefix), value, 0).Err()
}

func getKey(key string, keyPrefix string) string {
	return fmt.Sprintf("%s:%s", keyPrefix, key)
}
