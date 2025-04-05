package easy_ocr

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/minio/minio-go/v7"
	"github.com/satori/uuid"
	"purify/src/config"
	"purify/src/utils"
)

type EasyOcr struct {
	minioClient *minio.Client
	cfg         config.EasyOcrConfig
	minioCfg    config.MinioConfig
}

func NewEasyOcr(minioClient *minio.Client, cfg config.EasyOcrConfig, minioCfg config.MinioConfig) *EasyOcr {
	return &EasyOcr{
		minioClient: minioClient,
		cfg:         cfg,
		minioCfg:    minioCfg,
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

	easyOcrResp, err := processImage(imageBytes, e.cfg)
	if err != nil {
		utils.LogError(ctx, err, "failed to process image")
		http.Error(w, utils.Internal, http.StatusInternalServerError)
		return
	}

	if easyOcrResp.BlurredImage == "" {
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

	resp := ProcessImageResponse{Image: fmt.Sprintf("/%s/%s", e.minioCfg.BucketName, objectName)}
	if err = json.NewEncoder(w).Encode(resp); err != nil {
		utils.LogError(ctx, err, utils.MsgErrMarshalResponse)
		http.Error(w, utils.Internal, http.StatusInternalServerError)
		return
	}
}
