package chatgpt

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/pkg/errors"
	"purify/src/config"
)

var client = http.DefaultClient

func SendRequest(prompt string, cfg config.AIConfig) (string, error) {
	url := fmt.Sprintf("%s%s", cfg.BaseURL, cfg.CompletionsURL)
	requestBody := makeRequestData(cfg.Model, prompt)

	req, err := http.NewRequest(http.MethodPost, url, strings.NewReader(requestBody))
	if err != nil {
		return "", errors.Wrap(err, "failed to create request")
	}

	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", os.Getenv("CHATGPT_API_KEY")))
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return "", errors.Wrap(err, "failed to send request")
	}

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", errors.Wrap(err, "failed to read response body")
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", errors.Errorf("status code: %d, respBody: %s", resp.StatusCode, string(respBody))
	}

	var result respChoices
	if err = json.Unmarshal(respBody, &result); err != nil {
		return "", errors.Wrap(err, "failed to unmarshal response body")
	}

	for _, choice := range result.Choices {
		if choice.Index == 0 {
			return choice.Message.Content, nil
		}
	}

	return "", errors.Errorf("no answer found in response")
}

func makeRequestData(model string, prompt string) string {
	return fmt.Sprintf(`{"model":"%s","store":true,"messages":[{"role":"user","content":"%s"}]}`, model, prompt)
}

type respChoices struct {
	Choices []respChoice `json:"choices"`
}

type respChoice struct {
	Index   int         `json:"index"`
	Message respMessage `json:"message"`
}

type respMessage struct {
	Content string `json:"content"`
}
