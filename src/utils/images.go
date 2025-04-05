package utils

import (
	"bytes"
	"context"
	"io"
	"net/http"

	"github.com/minio/minio-go/v7"
	"github.com/pkg/errors"
)

func DownloadImage(url string) ([]byte, error) {
	resp, err := http.Get(url)
	if err != nil {
		return nil, errors.Wrap(err, "failed to get image")
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, errors.Wrap(err, "failed to read image")
	}

	return respBody, nil
}

func UploadImage(ctx context.Context, minioClient *minio.Client, bucketName, objectName string, data []byte) error {
	reader := bytes.NewReader(data)

	_, err := minioClient.PutObject(ctx, bucketName, objectName, reader, int64(len(data)), minio.PutObjectOptions{
		ContentType: "image/png",
	})
	if err != nil {
		return errors.Wrap(err, "failed to put object to bucket")
	}

	return nil
}
