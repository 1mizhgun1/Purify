package easy_ocr

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/pkg/errors"
	"purify/src/config"
)

var client = &http.Client{
	Timeout: 120 * time.Second,
}

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

type processImagesRequest struct {
	Images []string `json:"images"`
}

type processImagesResponse struct {
	Results []resItem `json:"results"`
}

type resItem struct {
	BlurredImage string `json:"blurred_image"`
}

func processImages(images [][]byte, cfg config.EasyOcrConfig) (processImagesResponse, error) {
	url := fmt.Sprintf("http://%s:%s/%s", cfg.Host, cfg.Port, cfg.EndpointParallel)

	reqImages := make([]string, 0, len(images))
	for _, image := range images {
		reqImages = append(reqImages, base64.StdEncoding.EncodeToString(image))
	}

	requestBody := processImagesRequest{Images: reqImages}
	requestBodyBytes, err := json.Marshal(requestBody)
	if err != nil {
		return processImagesResponse{}, errors.Wrap(err, "failed to marshal request body")
	}

	fmt.Println(string(requestBodyBytes))

	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(requestBodyBytes))
	if err != nil {
		return processImagesResponse{}, errors.Wrap(err, "failed to create request")
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return processImagesResponse{}, errors.Wrap(err, "failed to send request")
	}

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return processImagesResponse{}, errors.Wrap(err, "failed to read response body")
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return processImagesResponse{}, errors.Errorf("status code: %d, respBody: %s", resp.StatusCode, string(respBody))
	}

	var result processImagesResponse
	if err = json.Unmarshal(respBody, &result); err != nil {
		return processImagesResponse{}, errors.Wrap(err, "failed to unmarshal response body")
	}

	return result, nil
}
