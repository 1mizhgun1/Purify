package mistral_ai

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

func sendMistralAIRequest(prompt string, cfg config.MistralAIConfig) (string, error) {
	url := fmt.Sprintf("%s%s", cfg.BaseURL, cfg.CompletionsURL)
	requestBody := fmt.Sprintf(requestTemplate, prompt)

	req, err := http.NewRequest(http.MethodPost, url, strings.NewReader(requestBody))
	if err != nil {
		return "", errors.Wrap(err, "failed to create request")
	}

	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", os.Getenv("MISTRAL_AI_API_KEY")))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return "", errors.Wrap(err, "failed to send request")
	}

	if resp.StatusCode != http.StatusOK {
		return "", errors.Errorf("status code: %d", resp.StatusCode)
	}

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", errors.Wrap(err, "failed to read response body")
	}
	defer resp.Body.Close()

	var result MistralAIResponse
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
