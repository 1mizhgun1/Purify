package chatgpt

import (
	"context"
	"encoding/json"
	goerrors "errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"purify/src/cache"
	"purify/src/config"
	"purify/src/utils"

	"github.com/pkg/errors"
)

const (
	featureBlur    = "blur"
	featureReplace = "replace"
)

var (
	blurPreconceptionPrompt = "Найди в тексте все слова, словосочетания и предложения, которые содержат предвзятость. Результат представь в виде json, формат такой: {\\\"preconception\\\":[]}, в ответе напиши только результат и ничего лишнего. Текст для анализа: %s"
	blurAgitationPrompt     = "Найди в тексте все слова, словосочетания и предложения, которые содержат негативную агитацию. Результат представь в виде json, формат такой: {\\\"agitation\\\":[]}, в ответе напиши только результат и ничего лишнего. Текст для анализа: %s"
	blurAllPrompt           = "Найди в тексте все слова, словосочетания и предложения, которые содержат предвзятость или негативную агитацию. Результат представь в виде json, содержащего два массива строк, формат такой: {\\\"preconception\\\":[],\\\"agitation\\\":[]}, в ответе напиши только результат и ничего лишнего. Текст для анализа: %s"

	changePreconceptionPrompt = "Найди в тексте негативную предвзятость и скажи, на что её исправить. Все остальные негативные моменты, не связанные с предвзятостью, исправлять не нужно. Если в тексте нет предвзятости, то не нужно исправлять ничего. В ответе напиши только результат, что на что нужно заменить в тексте, и ничего лишнего. Формат ответа: [{\\\"from\\\":\\\"sometext\\\",\\\"to\\\":\\\"othertext\\\"}] Текст: %s"
	changeAgitationPrompt     = "Найди в тексте негативную агитацию и скажи, на что её исправить. Все остальные негативные моменты, не связанные с агитацией, исправлять не нужно. Если в тексте нет агитации, то не нужно исправлять ничего. В ответе напиши только результат, что на что нужно заменить в тексте, и ничего лишнего. Формат ответа: [{\\\"from\\\":\\\"sometext\\\",\\\"to\\\":\\\"othertext\\\"}] Текст: %s"
	changeAllPrompt           = "Найди в тексте негативную предвзятость и негативную агитацию и скажи, на что их исправить. Все остальные негативные моменты, не связанные с предвзятостью или агитацией, исправлять не нужно. Если в тексте нет предвзятости и агитации, то не нужно исправлять ничего. В ответе напиши только результат что на что нужно заменить в тексте, и ничего лишнего. Формат ответа: [{\\\"from\\\":\\\"sometext\\\",\\\"to\\\":\\\"othertext\\\",\\\"type\\\":1}] где type=1 - замена предвзятости, type=2 - замена агитации Текст: %s"
)

type ChatGPT struct {
	cfg   config.ChatGPTConfig
	cache *cache.Cache
}

func NewChatGPT(cfg config.ChatGPTConfig, cache *cache.Cache) *ChatGPT {
	return &ChatGPT{cfg: cfg, cache: cache}
}

type BlurRequest struct {
	Text          string `json:"text"`
	Preconception bool   `json:"preconception"`
	Agitation     bool   `json:"agitation"`
}

type ChangeRequest struct {
	Text          string `json:"text"`
	Preconception bool   `json:"preconception"`
	Agitation     bool   `json:"agitation"`
}

type BlurResponse struct {
	Preconception []string `json:"preconception"`
	Agitation     []string `json:"agitation"`
}

type ChangeResponse struct {
	Result []Replacement `json:"result"`
}

type Replacement struct {
	From string `json:"from"`
	To   string `json:"to"`
	Type int    `json:"type"`
}

func (c *ChatGPT) Blur(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()

	var req BlurRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		utils.LogError(ctx, err, utils.MsgErrUnmarshalRequest)
		http.Error(w, utils.Invalid, http.StatusBadRequest)
		return
	}

	var promptFormat string
	if req.Preconception && req.Agitation {
		promptFormat = blurAllPrompt
	} else if req.Preconception {
		promptFormat = blurPreconceptionPrompt
	} else if req.Agitation {
		promptFormat = blurAgitationPrompt
	} else {
		if err := json.NewEncoder(w).Encode(BlurResponse{}); err != nil {
			utils.LogError(ctx, err, utils.MsgErrMarshalResponse)
			http.Error(w, utils.Internal, http.StatusInternalServerError)
		}
		return
	}

	chunks := utils.SplitTextIntoChunks(req.Text, c.cfg.WordsInChunk, c.cfg.MaxChunks)

	wg := &sync.WaitGroup{}
	responsesCh := make(chan ChunkInChannel)
	errorsCh := make(chan error)

	for i, chunk := range chunks {
		wg.Add(1)
		time.Sleep(time.Millisecond)
		go c.sendChunkRequest(ctx, promptFormat, chunk, i, wg, responsesCh, errorsCh)
	}

	go func() {
		wg.Wait()
		close(responsesCh)
		close(errorsCh)
	}()

	chunkResponses := make([]ChunkInChannel, 0)
	chunkErrors := make([]error, 0)

	for {
		select {
		case response, ok := <-responsesCh:
			if !ok {
				goto end
			}
			chunkResponses = append(chunkResponses, response)
		case err, ok := <-errorsCh:
			if !ok {
				continue
			}
			chunkErrors = append(chunkErrors, err)
		}
	}

end:
	resp := joinResponses(chunkResponses)
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		utils.LogError(ctx, err, utils.MsgErrMarshalResponse)
		http.Error(w, utils.Internal, http.StatusInternalServerError)
		return
	}
}

type ChunkInChannel struct {
	Index int
	BlurResponse
}

func (c *ChatGPT) sendChunkRequest(ctx context.Context, promptFormat string, chunk string, index int, wg *sync.WaitGroup, responses chan<- ChunkInChannel, errorsCh chan<- error) {
	defer wg.Done()

	answerFromCache, err := c.cache.GetAnswer(ctx, chunk, featureBlur)
	if err != nil {
		if !goerrors.Is(err, cache.ErrNoAnswer) {
			errorsCh <- errors.Wrapf(err, "failed to get answer from cache, index=%d", index)
		} else {
			fmt.Printf("[DEBUG] no answer in cache, index=%d\n", index)
		}
	} else {
		fmt.Printf("[DEBUG] answer from cache, index=%d\n]", index)
		responses <- ChunkInChannel{Index: index, BlurResponse: BlurResponse{
			Preconception: answerFromCache.Preconception,
			Agitation:     answerFromCache.Agitation,
		}}
		return
	}

	prompt := fmt.Sprintf(promptFormat, chunk)
	answer, err := sendRequest(prompt, c.cfg)
	if err != nil {
		errorsCh <- errors.Wrapf(err, "failed to send chunk request, index=%d", index)
		return
	}

	answer = strings.TrimPrefix(answer, "```")
	answer = strings.TrimPrefix(answer, "json")
	answer = strings.TrimSuffix(answer, "```")

	fmt.Printf("[DEBUG] i=%d answer=%s\n", index, answer)

	var resp BlurResponse
	if err = json.Unmarshal([]byte(answer), &resp); err != nil {
		errorsCh <- errors.Wrapf(err, "failed to unmarshal chunk response, index=%d", index)
		return
	}

	responses <- ChunkInChannel{Index: index, BlurResponse: resp}

	if err = c.cache.SetAnswer(ctx, chunk, cache.Value{Preconception: resp.Preconception, Agitation: resp.Agitation}, featureBlur); err != nil {
		errorsCh <- errors.Wrapf(err, "failed to set answer in cache, index=%d", index)
	}
}

func joinResponses(responses []ChunkInChannel) BlurResponse {
	resp := BlurResponse{Preconception: make([]string, 0), Agitation: make([]string, 0)}
	for _, response := range responses {
		resp.Preconception = append(resp.Preconception, response.BlurResponse.Preconception...)
		resp.Agitation = append(resp.Agitation, response.BlurResponse.Agitation...)
	}

	return resp
}

func (c *ChatGPT) Replace(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()

	var req ChangeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		utils.LogError(ctx, err, utils.MsgErrUnmarshalRequest)
		http.Error(w, utils.Invalid, http.StatusBadRequest)
		return
	}

	// 0 - оба параметра, 1 - предвзятость, 2 - агитация
	requestType := 0

	var promptFormat string
	if req.Preconception && req.Agitation {
		promptFormat = changeAllPrompt
	} else if req.Preconception {
		promptFormat = changePreconceptionPrompt
		requestType = 1
	} else if req.Agitation {
		promptFormat = changeAgitationPrompt
		requestType = 2
	} else {
		if err := json.NewEncoder(w).Encode(BlurResponse{}); err != nil {
			utils.LogError(ctx, err, utils.MsgErrMarshalResponse)
			http.Error(w, utils.Internal, http.StatusInternalServerError)
		}
		return
	}

	prompt := fmt.Sprintf(promptFormat, req.Text)
	answer, err := sendRequest(prompt, c.cfg)
	if err != nil {
		utils.LogError(ctx, err, utils.MsgErrMarshalResponse)
		http.Error(w, utils.Internal, http.StatusInternalServerError)
		return
	}

	answer = strings.TrimPrefix(answer, "```")
	answer = strings.TrimPrefix(answer, "json")
	answer = strings.TrimSuffix(answer, "```")

	var resp ChangeResponse
	if err = json.Unmarshal([]byte(answer), &resp.Result); err != nil {
		utils.LogError(ctx, err, "invalid answer format from ChatGPT")
		http.Error(w, utils.Internal, http.StatusInternalServerError)
		return
	}

	if requestType != 0 {
		for i := range resp.Result {
			resp.Result[i].Type = requestType
		}
	}

	if err = json.NewEncoder(w).Encode(resp); err != nil {
		utils.LogError(ctx, err, utils.MsgErrMarshalResponse)
		http.Error(w, utils.Internal, http.StatusInternalServerError)
		return
	}
}
