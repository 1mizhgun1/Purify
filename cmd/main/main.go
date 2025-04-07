package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"

	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
	"purify/src/ai"
	"purify/src/cache"
	"purify/src/config"
	"purify/src/easy_ocr"
	"purify/src/middleware"
	"purify/src/mistral_ai"

	"github.com/gorilla/mux"
	"github.com/joho/godotenv"
	"github.com/pkg/errors"
	"github.com/redis/go-redis/v9"
)

func init() {
	if err := godotenv.Load(); err != nil {
		log.Fatalf("failed to load .env file: %v", err)
	}
}

func initMinio(minioClient *minio.Client, bucketName string, region string, logger *slog.Logger) error {
	ctx := context.Background()

	exists, err := minioClient.BucketExists(ctx, bucketName)
	if err != nil {
		return errors.Wrap(err, "failed to check bucket existence")
	}

	if !exists {
		if err = minioClient.MakeBucket(ctx, bucketName, minio.MakeBucketOptions{Region: region}); err != nil {
			return errors.Wrap(err, "failed to create bucket")
		}
		logger.Info("minio bucket created")

		if err = minioClient.SetBucketPolicy(ctx, bucketName, fmt.Sprintf(`{
			"Version":"2012-10-17",
			"Statement":[
				{
					"Effect":"Allow",
					"Principal":{"AWS":["*"]},
					"Action":["s3:GetObject"],
					"Resource":["arn:aws:s3:::%s/*"]
				}
			]
		}`, bucketName),
		); err != nil {
			return errors.Wrap(err, "failed to set bucket policy")
		}
		logger.Info("minio bucket policy set")

	} else {
		logger.Info("minio bucket already exists")
	}

	return nil
}

func main() {
	// =================================================================================================================
	// logger

	logFile, err := os.OpenFile(os.Getenv("MAIN_LOG_FILE"), os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0644)
	if err != nil {
		log.Fatalf("failed to open log file: %v", err)
	}
	defer logFile.Close()

	logger := slog.New(slog.NewJSONHandler(io.MultiWriter(logFile, os.Stdout), &slog.HandlerOptions{Level: slog.LevelInfo}))

	// logger
	// =================================================================================================================
	// config

	cfg := config.MustLoadConfig(os.Getenv("CONFIG_FILE"), logger)
	logger.Info("Config file loaded")

	// config
	// =================================================================================================================
	// redis

	redisOpts, err := redis.ParseURL(os.Getenv("REDIS_URL"))
	if err != nil {
		logger.Error(errors.Wrap(err, "failed to parse redis url").Error())
		return
	}
	redisClient := redis.NewClient(redisOpts)

	if err = redisClient.Ping(context.Background()).Err(); err != nil {
		logger.Error(errors.Wrap(err, "failed to ping redis").Error())
		return
	}
	logger.Info("Redis connected")

	// redis
	// =================================================================================================================
	// minio

	endpoint := os.Getenv("MINIO_ENDPOINT")
	accessKeyID := os.Getenv("MINIO_ROOT_USER")
	secretAccessKey := os.Getenv("MINIO_ROOT_PASSWORD")
	region := os.Getenv("MINIO_REGION")

	minioClient, err := minio.New(endpoint, &minio.Options{
		Creds:  credentials.NewStaticV4(accessKeyID, secretAccessKey, ""),
		Secure: false,
		Region: region,
	})
	if err != nil {
		logger.Error(errors.Wrap(err, "failed to initialize minio client").Error())
		return
	}
	logger.Info("Minio connected")

	if err = initMinio(minioClient, cfg.Minio.BucketName, region, logger); err != nil {
		logger.Error(errors.Wrap(err, "minio init error").Error())
		return
	}

	// minio
	// =================================================================================================================
	// entities

	redisCache := cache.NewCache(redisClient)

	mistralAI := mistral_ai.NewMistralAI(cfg.MistralAI)
	chatGPT := ai.NewAI(cfg.ChatGPT, redisCache, ai.TypeChatGPT)
	deepseek := ai.NewAI(cfg.Deepseek, redisCache, ai.TypeDeepseek)

	easyOcr := easy_ocr.NewEasyOcr(minioClient, cfg.EasyOcr, cfg.Minio, redisCache)

	// entities
	// =================================================================================================================
	// router

	reqIDMiddleware := middleware.CreateRequestIDMiddleware(logger)

	r := mux.NewRouter().PathPrefix("/api/v1").Subrouter()
	r.Use(reqIDMiddleware, middleware.CorsMiddleware, middleware.RecoverMiddleware)

	r.NotFoundHandler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	})

	r.Handle("/ping", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.Write([]byte("pong")) }))

	r.Handle("/analyze_text", http.HandlerFunc(mistralAI.AnalyzeText)).Methods(http.MethodPost, http.MethodOptions)

	r.Handle("/blur", http.HandlerFunc(chatGPT.Blur)).Methods(http.MethodPost, http.MethodOptions)
	r.Handle("/replace", http.HandlerFunc(chatGPT.Replace)).Methods(http.MethodPost, http.MethodOptions)

	r.Handle("/deepseek/blur", http.HandlerFunc(deepseek.Blur)).Methods(http.MethodPost, http.MethodOptions)
	r.Handle("/deepseek/replace", http.HandlerFunc(deepseek.Replace)).Methods(http.MethodPost, http.MethodOptions)

	r.Handle("/process_image", http.HandlerFunc(easyOcr.ProcessImage)).Methods(http.MethodPost, http.MethodOptions)

	http.Handle("/", r)
	server := http.Server{
		Handler:           middleware.PathMiddleware(r),
		Addr:              fmt.Sprintf(":%s", cfg.Main.Port),
		ReadTimeout:       cfg.Main.ReadTimeout,
		WriteTimeout:      cfg.Main.WriteTimeout,
		ReadHeaderTimeout: cfg.Main.ReadHeaderTimeout,
		IdleTimeout:       cfg.Main.IdleTimeout,
	}

	// router
	// =================================================================================================================
	// graceful shutdown

	signalCh := make(chan os.Signal, 1)
	signal.Notify(signalCh, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		if err = server.ListenAndServe(); err != nil {
			logger.Info("Server stopped")
		}
	}()
	logger.Info("Server started")

	sig := <-signalCh
	logger.Info("Received signal: " + sig.String())

	ctx, cancel := context.WithTimeout(context.Background(), cfg.Main.ShutdownTimeout)
	defer cancel()

	if err = server.Shutdown(ctx); err != nil {
		logger.Error(errors.Wrap(err, "failed to gracefully shutdown").Error())
	}

	// graceful shutdown
	// =================================================================================================================
}
