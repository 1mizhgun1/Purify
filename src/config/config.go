package config

import (
	"fmt"
	"log/slog"
	"os"
	"time"

	"gopkg.in/yaml.v3"
)

type LoggerKey string

const LoggerContextKey LoggerKey = "logger"

type Config struct {
	Main      MainConfig      `yaml:"main"`
	MistralAI MistralAIConfig `yaml:"mistral_ai"`
	ChatGPT   AIConfig        `yaml:"chat_gpt"`
	Deepseek  AIConfig        `yaml:"deepseek"`
	Minio     MinioConfig     `yaml:"minio"`
	EasyOcr   EasyOcrConfig   `yaml:"easy_ocr"`
}

type MainConfig struct {
	Port              string        `yaml:"port"`
	ReadTimeout       time.Duration `yaml:"read_timeout"`
	WriteTimeout      time.Duration `yaml:"write_timeout"`
	ReadHeaderTimeout time.Duration `yaml:"read_header_timeout"`
	IdleTimeout       time.Duration `yaml:"idle_timeout"`
	ShutdownTimeout   time.Duration `yaml:"shutdown_timeout"`
}

type MistralAIConfig struct {
	BaseURL        string `yaml:"base_url"`
	CompletionsURL string `yaml:"completions_url"`
	WordsInChunk   int    `yaml:"words_in_chunk"`
	MaxChunks      int    `yaml:"max_chunks"`
}

type AIConfig struct {
	BaseURL                  string `yaml:"base_url"`
	CompletionsURL           string `yaml:"completions_url"`
	Model                    string `yaml:"model"`
	WordsInChunk             int    `yaml:"words_in_chunk"`      // for blur
	MaxChunks                int    `yaml:"max_chunks"`          // for blur
	MaxTokensInChunk         int    `yaml:"max_tokens_in_chunk"` // for replace
	SimplifyMinTokensInChunk int    `yaml:"simplify_min_tokens_in_chunk"`
	SimplifyMaxTokensInChunk int    `yaml:"simplify_max_tokens_in_chunk"`
}

type MinioConfig struct {
	BucketName string `yaml:"bucket_name"`
}

type EasyOcrConfig struct {
	Host             string `yaml:"host"`
	Port             string `yaml:"port"`
	Endpoint         string `yaml:"endpoint"`
	EndpointParallel string `yaml:"endpoint_parallel"`
}

func MustLoadConfig(path string, logger *slog.Logger) *Config {
	cfg := &Config{}

	file, err := os.Open(path)
	if err != nil {
		logger.Error(fmt.Sprintf("failed to open config file: %v", err))
		return &Config{}
	}
	defer file.Close()

	if err = yaml.NewDecoder(file).Decode(cfg); err != nil {
		logger.Error(fmt.Sprintf("failed to decode config file: %v", err))
		return &Config{}
	}

	return cfg
}
