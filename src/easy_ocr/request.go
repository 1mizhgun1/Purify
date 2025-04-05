package easy_ocr

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/pkg/errors"
	"purify/src/config"
)

var client = http.DefaultClient

type processImageResponse struct {
	BlurredImage string `json:"blurred_image"`
}

func processImage(image []byte, cfg config.EasyOcrConfig) (processImageResponse, error) {
	url := fmt.Sprintf("http://%s:%s/%s", cfg.Host, cfg.Port, cfg.Endpoint)

	base64Image := base64.StdEncoding.EncodeToString(image)
	requestBody := fmt.Sprintf(`{"image":"%s"}`, base64Image)

	req, err := http.NewRequest(http.MethodPost, url, strings.NewReader(requestBody))
	if err != nil {
		return processImageResponse{}, errors.Wrap(err, "failed to create request")
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return processImageResponse{}, errors.Wrap(err, "failed to send request")
	}

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return processImageResponse{}, errors.Wrap(err, "failed to read response body")
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return processImageResponse{}, errors.Errorf("status code: %d, respBody: %s", resp.StatusCode, string(respBody))
	}

	var result processImageResponse
	if err = json.Unmarshal(respBody, &result); err != nil {
		return processImageResponse{}, errors.Wrap(err, "failed to unmarshal response body")
	}

	return result, nil
}
