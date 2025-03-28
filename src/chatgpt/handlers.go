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

	changePreconceptionPrompt = "Найди в тексте негативную предвзятость и скажи, на что её исправить. При этом не нужно менять смысл на положительный, нужно лишь сгладить негатив. Все остальные негативные моменты, не связанные с предвзятостью, исправлять не нужно. Если в тексте нет предвзятости, то не нужно исправлять ничего. В ответе напиши только результат, что на что нужно заменить в тексте, и ничего лишнего. Формат ответа: [{\\\"from\\\":\\\"sometext\\\",\\\"to\\\":\\\"othertext\\\"}] Текст: %s"
	changeAgitationPrompt     = "Найди в тексте негативную агитацию и скажи, на что её исправить. При этом не нужно менять смысл на положительный, нужно лишь сгладить негатив. Все остальные негативные моменты, не связанные с агитацией, исправлять не нужно. Если в тексте нет агитации, то не нужно исправлять ничего. В ответе напиши только результат, что на что нужно заменить в тексте, и ничего лишнего. Формат ответа: [{\\\"from\\\":\\\"sometext\\\",\\\"to\\\":\\\"othertext\\\"}] Текст: %s"
	changeAllPrompt           = "Найди в тексте негативную предвзятость и негативную агитацию и скажи, на что их исправить. При этом не нужно менять смысл на положительный, нужно лишь сгладить негатив. Все остальные негативные моменты, не связанные с предвзятостью или агитацией, исправлять не нужно. Если в тексте нет предвзятости и агитации, то не нужно исправлять ничего. В ответе напиши только результат что на что нужно заменить в тексте, и ничего лишнего. Формат ответа: [{\\\"from\\\":\\\"sometext\\\",\\\"to\\\":\\\"othertext\\\",\\\"type\\\":1}] где type=1 - замена предвзятости, type=2 - замена агитации Текст: %s"
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

type BlurResponse struct {
	Preconception []string `json:"preconception"`
	Agitation     []string `json:"agitation"`
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

	// 0 - оба параметра, 1 - предвзятость, 2 - агитация
	requestType := 0

	var promptFormat string
	if req.Preconception && req.Agitation {
		promptFormat = blurAllPrompt
	} else if req.Preconception {
		promptFormat = blurPreconceptionPrompt
		requestType = 1
	} else if req.Agitation {
		promptFormat = blurAgitationPrompt
		requestType = 2
	} else {
		if err := json.NewEncoder(w).Encode(BlurResponse{}); err != nil {
			utils.LogError(ctx, err, utils.MsgErrMarshalResponse)
			http.Error(w, utils.Internal, http.StatusInternalServerError)
		}
		return
	}

	chunks := utils.SplitTextIntoChunks(req.Text, c.cfg.WordsInChunk, c.cfg.MaxChunks)

	wg := &sync.WaitGroup{}
	responsesCh := make(chan ChunkInChannelBlur)
	errorsCh := make(chan error)

	for i, chunk := range chunks {
		wg.Add(1)
		time.Sleep(time.Millisecond)
		go c.sendChunkRequestBlur(ctx, promptFormat, requestType, chunk, i, wg, responsesCh, errorsCh)
	}

	go func() {
		wg.Wait()
		close(responsesCh)
		close(errorsCh)
	}()

	chunkResponses := make([]ChunkInChannelBlur, 0)
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
	resp := joinResponsesBlur(chunkResponses)
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		utils.LogError(ctx, err, utils.MsgErrMarshalResponse)
		http.Error(w, utils.Internal, http.StatusInternalServerError)
		return
	}
}

type ChunkInChannelBlur struct {
	Index int
	BlurResponse
}

func (c *ChatGPT) sendChunkRequestBlur(ctx context.Context, promptFormat string, requestType int, chunk string, index int, wg *sync.WaitGroup, responses chan<- ChunkInChannelBlur, errorsCh chan<- error) {
	defer wg.Done()

	answerFromCacheString, err := c.cache.GetAnswer(ctx, chunk, fmt.Sprintf("%s%d", featureBlur, requestType))
	if err != nil {
		if !goerrors.Is(err, cache.ErrNoAnswer) {
			errorsCh <- errors.Wrapf(err, "failed to get answer from cache, index=%d", index)
		} else {
			fmt.Printf("[DEBUG] no answer in cache, index=%d\n", index)
		}
	} else {
		fmt.Printf("[DEBUG] answer from cache, index=%d\n]", index)

		var answerFromCache BlurResponse
		if err = json.Unmarshal([]byte(answerFromCacheString), &answerFromCache); err != nil {
			errorsCh <- errors.Wrapf(err, "failed to unmarshal answer from cache, index=%d", index)
		}

		responses <- ChunkInChannelBlur{Index: index, BlurResponse: answerFromCache}
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

	responses <- ChunkInChannelBlur{Index: index, BlurResponse: resp}

	respBytes, err := json.Marshal(resp)
	if err != nil {
		errorsCh <- errors.Wrapf(err, "failed to marshal chunk response, index=%d", index)
	}

	if err = c.cache.SetAnswer(ctx, chunk, string(respBytes), fmt.Sprintf("%s%d", featureBlur, requestType)); err != nil {
		errorsCh <- errors.Wrapf(err, "failed to set answer in cache, index=%d", index)
	}
}

func joinResponsesBlur(responses []ChunkInChannelBlur) BlurResponse {
	resp := BlurResponse{Preconception: make([]string, 0), Agitation: make([]string, 0)}
	for _, response := range responses {
		resp.Preconception = append(resp.Preconception, response.BlurResponse.Preconception...)
		resp.Agitation = append(resp.Agitation, response.BlurResponse.Agitation...)
	}

	return resp
}

// BLUR ZONE
// =====================================================================================================================
// REPLACE ZONE

type ReplaceRequest struct {
	Blocks        []string `json:"blocks"`
	Preconception bool     `json:"preconception"`
	Agitation     bool     `json:"agitation"`
}

type ReplaceResponse struct {
	Blocks []Block `json:"blocks"`
}

type Block struct {
	Text string `json:"text"`
	ReplaceResponseGPT
}

type ReplaceResponseGPT struct {
	Result []Replacement `json:"result"`
}

func (c *ChatGPT) Replace(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()

	var req ReplaceRequest
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
		resp := ReplaceResponse{Blocks: make([]Block, 0)}
		for _, block := range req.Blocks {
			resp.Blocks = append(resp.Blocks, Block{Text: block, ReplaceResponseGPT: ReplaceResponseGPT{Result: make([]Replacement, 0)}})
		}

		if err := json.NewEncoder(w).Encode(resp); err != nil {
			utils.LogError(ctx, err, utils.MsgErrMarshalResponse)
			http.Error(w, utils.Internal, http.StatusInternalServerError)
		}
		return
	}

	chunks := utils.SplitBlocksIntoChunks(req.Blocks, c.cfg.MaxTokensInChunk)

	wg := &sync.WaitGroup{}
	responsesCh := make(chan ChunkInChannelReplace)
	errorsCh := make(chan error)

	for i, chunk := range chunks {
		wg.Add(1)
		time.Sleep(time.Millisecond)
		go c.sendChunkRequestReplace(ctx, promptFormat, requestType, chunk, i, wg, responsesCh, errorsCh)
	}

	go func() {
		wg.Wait()
		close(responsesCh)
		close(errorsCh)
	}()

	chunkResponses := make([]ChunkInChannelReplace, 0)
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
	fmt.Printf("[DEBUG] chunkErrors: %v\n", chunkErrors)

	allResponses := joinResponsesReplace(chunkResponses)

	resp := ReplaceResponse{Blocks: make([]Block, 0)}
	for _, block := range req.Blocks {
		found := make([]Replacement, 0)
		for _, replacement := range allResponses.Result {
			if strings.Contains(block, replacement.From) {
				found = append(found, replacement)
			}
		}
		resp.Blocks = append(resp.Blocks, Block{Text: block, ReplaceResponseGPT: ReplaceResponseGPT{Result: found}})
	}

	if err := json.NewEncoder(w).Encode(resp); err != nil {
		utils.LogError(ctx, err, utils.MsgErrMarshalResponse)
		http.Error(w, utils.Internal, http.StatusInternalServerError)
		return
	}
}

type ChunkInChannelReplace struct {
	Index int
	ReplaceResponseGPT
}

func (c *ChatGPT) sendChunkRequestReplace(ctx context.Context, promptFormat string, requestType int, chunk string, index int, wg *sync.WaitGroup, responses chan<- ChunkInChannelReplace, errorsCh chan<- error) {
	defer wg.Done()

	answerFromCacheString, err := c.cache.GetAnswer(ctx, chunk, fmt.Sprintf("%s%d", featureReplace, requestType))
	if err != nil {
		if !goerrors.Is(err, cache.ErrNoAnswer) {
			errorsCh <- errors.Wrapf(err, "failed to get answer from cache, index=%d", index)
		} else {
			fmt.Printf("[DEBUG] no answer in cache, index=%d\n", index)
		}
	} else {
		fmt.Printf("[DEBUG] answer from cache, index=%d\n]", index)

		var answerFromCache ReplaceResponseGPT
		if err = json.Unmarshal([]byte(answerFromCacheString), &answerFromCache.Result); err != nil {
			errorsCh <- errors.Wrapf(err, "failed to unmarshal answer from cache, index=%d", index)
		}

		responses <- ChunkInChannelReplace{Index: index, ReplaceResponseGPT: answerFromCache}
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

	var resp ReplaceResponseGPT
	if err = json.Unmarshal([]byte(answer), &resp.Result); err != nil {
		errorsCh <- errors.Wrapf(err, "failed to unmarshal chunk response, index=%d", index)
		return
	}
	if requestType != 0 {
		for i := range resp.Result {
			resp.Result[i].Type = requestType
		}
	}

	responses <- ChunkInChannelReplace{Index: index, ReplaceResponseGPT: resp}

	respBytes, err := json.Marshal(resp.Result)
	if err != nil {
		errorsCh <- errors.Wrapf(err, "failed to marshal chunk response, index=%d", index)
	}

	if err = c.cache.SetAnswer(ctx, chunk, string(respBytes), fmt.Sprintf("%s%d", featureReplace, requestType)); err != nil {
		errorsCh <- errors.Wrapf(err, "failed to set answer in cache, index=%d", index)
	}
}

func joinResponsesReplace(responses []ChunkInChannelReplace) ReplaceResponseGPT {
	resp := ReplaceResponseGPT{Result: make([]Replacement, 0)}
	for _, response := range responses {
		resp.Result = append(resp.Result, response.Result...)
	}
	return resp
}
