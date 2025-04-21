package easy_ocr

import (
	"encoding/base64"
	"encoding/json"
	goerrors "errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/minio/minio-go/v7"
	"github.com/satori/uuid"
	"purify/src/cache"
	"purify/src/config"
	"purify/src/utils"
)

const featureProcessImage = "process_image"

type EasyOcr struct {
	minioClient *minio.Client
	cfg         config.EasyOcrConfig
	minioCfg    config.MinioConfig
	cache       *cache.Cache
}

func NewEasyOcr(minioClient *minio.Client, cfg config.EasyOcrConfig, minioCfg config.MinioConfig, cache *cache.Cache) *EasyOcr {
	return &EasyOcr{
		minioClient: minioClient,
		cfg:         cfg,
		minioCfg:    minioCfg,
		cache:       cache,
	}
}

type ProcessImageRequest struct {
	Image string `json:"image"`
}

type ProcessImageResponse struct {
	Image string `json:"image"`
}

func (e *EasyOcr) ProcessImage(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()

	var req ProcessImageRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		utils.LogError(ctx, err, utils.MsgErrUnmarshalRequest)
		http.Error(w, utils.Invalid, http.StatusBadRequest)
		return
	}

	imageBytes, err := utils.DownloadImage(req.Image)
	if err != nil {
		utils.LogError(ctx, err, "failed to download image")
		http.Error(w, utils.Internal, http.StatusInternalServerError)
		return
	}

	answerFromCache, err := e.cache.GetAnswer(ctx, string(imageBytes), featureProcessImage)
	if err != nil {
		if !goerrors.Is(err, cache.ErrNoAnswer) {
			utils.LogError(ctx, err, "failed to get answer from cache")
		} else {
			fmt.Printf("[DEBUG] no answer in cache\n")
		}
	} else {
		fmt.Printf("[DEBUG] answer from cache\n")
		resp := ProcessImageResponse{Image: answerFromCache}
		if err = json.NewEncoder(w).Encode(resp); err != nil {
			utils.LogError(ctx, err, utils.MsgErrMarshalResponse)
			http.Error(w, utils.Internal, http.StatusInternalServerError)
		}
		return
	}

	easyOcrResp, err := processImage(imageBytes, e.cfg)
	if err != nil {
		utils.LogError(ctx, err, "failed to process image")
		http.Error(w, utils.Internal, http.StatusInternalServerError)
		return
	}

	if easyOcrResp.BlurredImage == "" {
		if err = e.cache.SetAnswer(ctx, string(imageBytes), "", featureProcessImage); err != nil {
			utils.LogError(ctx, err, "failed to cache answer")
		}

		resp := ProcessImageResponse{Image: ""}
		if err = json.NewEncoder(w).Encode(resp); err != nil {
			utils.LogError(ctx, err, utils.MsgErrMarshalResponse)
			http.Error(w, utils.Internal, http.StatusInternalServerError)
		}
		return
	}

	blurredImageBytes, err := base64.StdEncoding.DecodeString(easyOcrResp.BlurredImage)
	if err != nil {
		utils.LogError(ctx, err, "failed to decode blurred image")
		http.Error(w, utils.Internal, http.StatusInternalServerError)
		return
	}

	objectName := fmt.Sprintf("%s.png", uuid.NewV4().String())
	if err = utils.UploadImage(ctx, e.minioClient, e.minioCfg.BucketName, objectName, blurredImageBytes); err != nil {
		utils.LogError(ctx, err, "failed to upload image")
		http.Error(w, utils.Internal, http.StatusInternalServerError)
		return
	}

	url := fmt.Sprintf("/%s/%s", e.minioCfg.BucketName, objectName)
	if err = e.cache.SetAnswer(ctx, string(imageBytes), url, featureProcessImage); err != nil {
		utils.LogError(ctx, err, "failed to cache answer")
	}

	resp := ProcessImageResponse{Image: url}
	if err = json.NewEncoder(w).Encode(resp); err != nil {
		utils.LogError(ctx, err, utils.MsgErrMarshalResponse)
		http.Error(w, utils.Internal, http.StatusInternalServerError)
		return
	}
}

func (e *EasyOcr) SwapImage(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()

	w.Header().Del("Content-Type")
	w.Header().Set("Content-Type", "image/png")

	imageURL := r.URL.Query().Get("originalUrl")
	imageURL = strings.TrimLeft(imageURL, "\\")

	imageBytes, err := utils.DownloadImage(imageURL)
	if err != nil {
		utils.LogError(ctx, err, "failed to download image")
		w.WriteHeader(http.StatusInternalServerError)
		return
	}

	contentType := http.DetectContentType(imageBytes)
	if contentType == "image/svg+xml" {
		imageBytes, err = utils.SvgToPng(imageBytes)
		if err != nil {
			utils.LogError(ctx, err, "failed to convert svg to png")
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
	}

	answerFromCache, err := e.cache.GetAnswer(ctx, string(imageBytes), featureProcessImage)
	if err != nil {
		if !goerrors.Is(err, cache.ErrNoAnswer) {
			utils.LogError(ctx, err, "failed to get answer from cache")
		} else {
			fmt.Printf("[DEBUG] no answer in cache\n")
		}
	} else {
		fmt.Printf("[DEBUG] answer from cache\n")

		if answerFromCache != "" {
			answerFromCache = strings.TrimPrefix(answerFromCache, "/")
			parts := strings.Split(answerFromCache, "/")
			if len(parts) != 2 {
				utils.LogError(ctx, err, "failed to parse answer from cache")
				w.WriteHeader(http.StatusInternalServerError)
				return
			}

			imageBytes, err = utils.DownloadImageFromBucket(ctx, e.minioClient, e.minioCfg.BucketName, parts[1])
			if err != nil {
				utils.LogError(ctx, err, "failed to download image by cache URL")
				w.WriteHeader(http.StatusInternalServerError)
				return
			}
		}

		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(imageBytes)
		return
	}

	easyOcrResp, err := processImage(imageBytes, e.cfg)
	if err != nil {
		utils.LogError(ctx, err, "failed to process image")
		w.WriteHeader(http.StatusInternalServerError)
		return
	}

	if easyOcrResp.BlurredImage == "" {
		if err = e.cache.SetAnswer(ctx, string(imageBytes), "", featureProcessImage); err != nil {
			utils.LogError(ctx, err, "failed to cache answer")
		}

		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(imageBytes)
		return
	}

	blurredImageBytes, err := base64.StdEncoding.DecodeString(easyOcrResp.BlurredImage)
	if err != nil {
		utils.LogError(ctx, err, "failed to decode blurred image")
		w.WriteHeader(http.StatusInternalServerError)
		return
	}

	objectName := fmt.Sprintf("%s.png", uuid.NewV4().String())
	if err = utils.UploadImage(ctx, e.minioClient, e.minioCfg.BucketName, objectName, blurredImageBytes); err != nil {
		utils.LogError(ctx, err, "failed to upload image")
		w.WriteHeader(http.StatusInternalServerError)
		return
	}

	url := fmt.Sprintf("/%s/%s", e.minioCfg.BucketName, objectName)
	if err = e.cache.SetAnswer(ctx, string(imageBytes), url, featureProcessImage); err != nil {
		utils.LogError(ctx, err, "failed to cache answer")
	}

	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(blurredImageBytes)
}

type ProcessImagesRequest struct {
	Images []string `json:"images"`
}

type ProcessImagesResponse struct {
	Images map[string]string `json:"images"`
}

func (e *EasyOcr) ProcessImages(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()

	var req ProcessImagesRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		utils.LogError(ctx, err, utils.MsgErrUnmarshalRequest)
		http.Error(w, utils.Invalid, http.StatusBadRequest)
		return
	}

	resp := ProcessImagesResponse{Images: make(map[string]string)}

	easyOcrBatch := make([][]byte, 0)
	batchInfo := make([]string, 0)

	for _, image := range req.Images {
		imageBytes, err := utils.DownloadImage(image)
		if err != nil {
			utils.LogError(ctx, err, "failed to download image: "+image)
			resp.Images[image] = ""
			continue
		}

		answerFromCache, err := e.cache.GetAnswer(ctx, string(imageBytes), featureProcessImage)
		if err != nil {
			if !goerrors.Is(err, cache.ErrNoAnswer) {
				utils.LogError(ctx, err, "failed to get answer from cache")
			} else {
				fmt.Printf("[DEBUG] no answer in cache\n")
			}
			easyOcrBatch = append(easyOcrBatch, imageBytes)
			batchInfo = append(batchInfo, image)
		} else {
			fmt.Printf("[DEBUG] answer from cache\n")
			resp.Images[image] = answerFromCache
		}
	}

	easyOcrResp, err := processImages(easyOcrBatch, e.cfg)
	if err != nil {
		utils.LogError(ctx, err, "failed to process images")
		http.Error(w, utils.Internal, http.StatusInternalServerError)
		return
	}

	for i, res := range easyOcrResp.Results {
		blurredImageBytes, err := base64.StdEncoding.DecodeString(res.BlurredImage)
		if err != nil {
			utils.LogError(ctx, err, "failed to decode blurred image")
			resp.Images[batchInfo[i]] = ""
			continue
		}

		objectName := fmt.Sprintf("%s.png", uuid.NewV4().String())
		if err = utils.UploadImage(ctx, e.minioClient, e.minioCfg.BucketName, objectName, blurredImageBytes); err != nil {
			utils.LogError(ctx, err, "failed to upload image")
			http.Error(w, utils.Internal, http.StatusInternalServerError)
			resp.Images[batchInfo[i]] = ""
			continue
		}

		url := fmt.Sprintf("/%s/%s", e.minioCfg.BucketName, objectName)
		if err = e.cache.SetAnswer(ctx, string(easyOcrBatch[i]), url, featureProcessImage); err != nil {
			utils.LogError(ctx, err, "failed to cache answer")
		}

		resp.Images[batchInfo[i]] = url
	}

	if err = json.NewEncoder(w).Encode(resp); err != nil {
		utils.LogError(ctx, err, utils.MsgErrMarshalResponse)
		http.Error(w, utils.Internal, http.StatusInternalServerError)
		return
	}
}
