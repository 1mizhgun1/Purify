package ai

import (
	"context"
	"encoding/json"
	goerrors "errors"
	"fmt"
	"net/http"
	"strings"
	"sync"

	"purify/src/ai/chatgpt"
	"purify/src/ai/deepseek"
	"purify/src/cache"
	"purify/src/config"
	"purify/src/utils"

	"github.com/jackc/pgx/v4"
	"github.com/jackc/pgx/v4/pgxpool"
	"github.com/pkg/errors"
)

type AIType string

const (
	featureBlur     = "blur"
	featureReplace  = "replace"
	featureSimplify = "simplify"

	TypeChatGPT  AIType = "chatgpt"
	TypeDeepseek AIType = "deepseek"
)

var (
	blurPreconceptionPrompt = "Найди в тексте все слова, словосочетания и предложения, которые содержат предвзятость. Результат представь в виде json, формат такой: {\\\"preconception\\\":[]}, в ответе напиши только результат и ничего лишнего. Текст для анализа: %s"
	blurAgitationPrompt     = "Найди в тексте все слова, словосочетания и предложения, которые содержат негативную агитацию. Результат представь в виде json, формат такой: {\\\"agitation\\\":[]}, в ответе напиши только результат и ничего лишнего. Текст для анализа: %s"
	blurAllPrompt           = "Найди в тексте все слова, словосочетания и предложения, которые содержат предвзятость или негативную агитацию. Результат представь в виде json, содержащего два массива строк, формат такой: {\\\"preconception\\\":[],\\\"agitation\\\":[]}, в ответе напиши только результат и ничего лишнего. Текст для анализа: %s"

	changePreconceptionPrompt = "Найди в тексте негативную предвзятость и скажи, на что её исправить. При этом не нужно менять смысл на положительный, нужно лишь сгладить негатив. Все остальные негативные моменты, не связанные с предвзятостью, исправлять не нужно. Если в тексте нет предвзятости, то не нужно исправлять ничего. В ответе напиши только результат, что на что нужно заменить в тексте, и ничего лишнего. Формат ответа: [{\\\"from\\\":\\\"sometext\\\",\\\"to\\\":\\\"othertext\\\"}] Текст: %s"
	changeAgitationPrompt     = "Найди в тексте негативную агитацию и скажи, на что её исправить. При этом не нужно менять смысл на положительный, нужно лишь сгладить негатив. Все остальные негативные моменты, не связанные с агитацией, исправлять не нужно. Если в тексте нет агитации, то не нужно исправлять ничего. В ответе напиши только результат, что на что нужно заменить в тексте, и ничего лишнего. Формат ответа: [{\\\"from\\\":\\\"sometext\\\",\\\"to\\\":\\\"othertext\\\"}] Текст: %s"
	changeAllPrompt           = "Найди в тексте негативную предвзятость и негативную агитацию и скажи, на что их исправить. При этом не нужно менять смысл на положительный, нужно лишь сгладить негатив. Все остальные негативные моменты, не связанные с предвзятостью или агитацией, исправлять не нужно. Если в тексте нет предвзятости и агитации, то не нужно исправлять ничего. В ответе напиши только результат что на что нужно заменить в тексте, и ничего лишнего. Формат ответа: [{\\\"from\\\":\\\"sometext\\\",\\\"to\\\":\\\"othertext\\\",\\\"type\\\":1}] где type=1 - замена предвзятости, type=2 - замена агитации Текст: %s"

	simplifyTextPrompt = "Я даю тебе набор блоков текста, для каждого блока тебе нужно сделать его суммаризацию и упрощение одновременно (использовать более простую лексику, термины). Если текст не содержит смысла, либо представляет собой числа/формулы/html-код/заголовки, и вообще если текст не треубет суммаризации или упрощения - нужно оставить исходный текст без изменений. Формат входных данных: [\\\"текст 1\\\", \\\"текст 2\\\"] ; формат выходных даных: [{\\\"from\\\":\\\"sometext\\\",\\\"to\\\":\\\"othertext\\\"}] . То есть тебе нужно указать, какие блоки во что превратились. Напиши только ответ в заданном формате и ничего лишнего, используй обычные двойные кавычки, не пиши в ответе сложность, только текста. Входные данные (блоки) для работы: %s"
	analyzeTextPrompt  = "Я даю тебе набор блоков текста, для каждого блока тебе нужно определить общее количество слов и количество агрессивных слов, количество негативной агитации, количество нецензурной лексики, количество предвзятости. Если текст не содержит смысла, либо представляет собой числа/формулы/html-код/заголовки - не нужно учитывать его. Формат входных данных: [\\\"текст 1\\\", \\\"текст 2\\\"] ; формат выходных даных: {\\\"total_words\\\":10,\\\"aggressive_words\\\":1,\\\"aggitation_words\\\":0,\\\"mat_words\\\":1,\\\"bias_words\\\":0}. То есть тебе нужно указать количество слов в блоке и количество агрессивных слов, слов с негативной агитаций, с ненормативной лексикой, предвзятостью. Напиши только ответ в заданном формате и ничего лишнего, используй обычные двойные кавычки, пиши в ответе только числа. Входные данные (блоки) для работы: %s"

	replacerReq  = strings.NewReplacer(`"`, `\\\"`)
	replacerResp = strings.NewReplacer(`\"`, `"`)
)

type AI struct {
	cfg   config.AIConfig
	cache *cache.Cache
	db    *pgxpool.Pool
	kind  AIType
}

func NewAI(cfg config.AIConfig, cache *cache.Cache, db *pgxpool.Pool, kind AIType) *AI {
	return &AI{cfg: cfg, cache: cache, db: db, kind: kind}
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

func (a *AI) Blur(w http.ResponseWriter, r *http.Request) {
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

	chunks := utils.SplitTextIntoChunks(req.Text, a.cfg.WordsInChunk, a.cfg.MaxChunks)

	wg := &sync.WaitGroup{}
	responsesCh := make(chan ChunkInChannelBlur)
	errorsCh := make(chan error)

	for i, chunk := range chunks {
		wg.Add(1)
		go a.sendChunkRequestBlur(ctx, promptFormat, requestType, chunk, i, wg, responsesCh, errorsCh)
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

func (a *AI) sendChunkRequestBlur(ctx context.Context, promptFormat string, requestType int, chunk string, index int, wg *sync.WaitGroup, responses chan<- ChunkInChannelBlur, errorsCh chan<- error) {
	defer wg.Done()

	answerFromCacheString, err := a.cache.GetAnswer(ctx, chunk, fmt.Sprintf("%s%d", featureBlur, requestType))
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
	answer, err := a.SendRequest(prompt)
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

	if err = a.cache.SetAnswer(ctx, chunk, string(respBytes), fmt.Sprintf("%s%d", featureBlur, requestType)); err != nil {
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

func (a *AI) Replace(w http.ResponseWriter, r *http.Request) {
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

	chunks := utils.SplitBlocksIntoChunks(req.Blocks, a.cfg.MaxTokensInChunk)

	wg := &sync.WaitGroup{}
	responsesCh := make(chan ChunkInChannelReplace)
	errorsCh := make(chan error)

	for i, chunk := range chunks {
		wg.Add(1)
		go a.sendChunkRequestReplace(ctx, promptFormat, requestType, chunk, i, wg, responsesCh, errorsCh)
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

func (a *AI) sendChunkRequestReplace(ctx context.Context, promptFormat string, requestType int, chunk string, index int, wg *sync.WaitGroup, responses chan<- ChunkInChannelReplace, errorsCh chan<- error) {
	defer wg.Done()

	answerFromCacheString, err := a.cache.GetAnswer(ctx, chunk, fmt.Sprintf("%s%d", featureReplace, requestType))
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
	answer, err := a.SendRequest(prompt)
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

	if err = a.cache.SetAnswer(ctx, chunk, string(respBytes), fmt.Sprintf("%s%d", featureReplace, requestType)); err != nil {
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

func (a *AI) SendRequest(prompt string) (string, error) {
	var (
		answer string
		err    error
	)

	switch a.kind {
	case TypeChatGPT:
		answer, err = chatgpt.SendRequest(prompt, a.cfg)
	case TypeDeepseek:
		answer, err = deepseek.SendRequest(prompt, a.cfg)
	default:
		answer, err = "", fmt.Errorf("unknown AI type: %s", a.kind)
	}

	return answer, err
}

type SimplifyRequest struct {
	Blocks []string `json:"blocks"`
}

type SimplifyResponse struct {
	Result []SimplifyReplacement `json:"result"`
}

type SimplifyReplacement struct {
	From string `json:"from"`
	To   string `json:"to"`
}

func (a *AI) Simplify(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()

	var req SimplifyRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		utils.LogError(ctx, err, utils.MsgErrUnmarshalRequest)
		http.Error(w, utils.Invalid, http.StatusBadRequest)
		return
	}

	chunks := utils.SplitBlocks(req.Blocks, a.cfg.SimplifyMinTokensInChunk, a.cfg.SimplifyMaxTokensInChunk)

	wg := &sync.WaitGroup{}
	responsesCh := make(chan ChunkInChannelSimplify)
	errorsCh := make(chan error)

	for i, chunk := range chunks {
		wg.Add(1)
		go a.sendChunkRequestSimplify(ctx, simplifyTextPrompt, chunk, i, wg, responsesCh, errorsCh)
	}

	go func() {
		wg.Wait()
		close(responsesCh)
		close(errorsCh)
	}()

	chunkResponses := make([]ChunkInChannelSimplify, 0)
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

	resp := joinResponsesSimplify(chunkResponses)
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		utils.LogError(ctx, err, utils.MsgErrMarshalResponse)
		http.Error(w, utils.Internal, http.StatusInternalServerError)
		return
	}
}

type ChunkInChannelSimplify struct {
	SimplifyResponse
	Index int `json:"index"`
}

func (a *AI) sendChunkRequestSimplify(ctx context.Context, promptFormat string, chunkSlice []string, index int, wg *sync.WaitGroup, responses chan<- ChunkInChannelSimplify, errorsCh chan<- error) {
	defer wg.Done()

	defaultResp := ChunkInChannelSimplify{Index: index}

	chunkBytes, err := json.Marshal(chunkSlice)
	if err != nil {
		errorsCh <- errors.Wrapf(err, "failed to marshal chunk slice")
		responses <- defaultResp
	}
	chunk := string(chunkBytes)

	answerFromCacheString, err := a.cache.GetAnswer(ctx, chunk, featureSimplify)
	if err != nil {
		if !goerrors.Is(err, cache.ErrNoAnswer) {
			errorsCh <- errors.Wrapf(err, "failed to get answer from cache, index=%d", index)
		} else {
			fmt.Printf("[DEBUG] no answer in cache, index=%d\n", index)
		}
	} else {
		fmt.Printf("[DEBUG] answer from cache, index=%d\n]", index)

		var answerFromCache ChunkInChannelSimplify
		if err = json.Unmarshal([]byte(answerFromCacheString), &answerFromCache.Result); err != nil {
			errorsCh <- errors.Wrapf(err, "failed to unmarshal answer from cache, index=%d", index)
		}
		answerFromCache.Index = index

		responses <- answerFromCache
		return
	}

	chunkToAI := replacerReq.Replace(chunk)
	prompt := fmt.Sprintf(promptFormat, chunkToAI)
	answer, err := a.SendRequest(prompt)
	if err != nil {
		errorsCh <- errors.Wrapf(err, "failed to send chunk request, index=%d", index)
		return
	}

	answer = strings.TrimPrefix(answer, "```")
	answer = strings.TrimPrefix(answer, "json")
	answer = strings.TrimSuffix(answer, "```")
	answer = replacerResp.Replace(answer)

	fmt.Printf("[DEBUG] i=%d answer=%s\n", index, answer)

	var resp ChunkInChannelSimplify
	if err = json.Unmarshal([]byte(answer), &resp.Result); err != nil {
		errorsCh <- errors.Wrapf(err, "failed to unmarshal chunk response, index=%d", index)
		return
	}

	resp.Index = index
	responses <- resp

	respBytes, err := json.Marshal(resp.Result)
	if err != nil {
		errorsCh <- errors.Wrapf(err, "failed to marshal chunk response, index=%d", index)
	}

	if err = a.cache.SetAnswer(ctx, chunk, string(respBytes), featureSimplify); err != nil {
		errorsCh <- errors.Wrapf(err, "failed to set answer in cache, index=%d", index)
	}
}

func joinResponsesSimplify(responses []ChunkInChannelSimplify) SimplifyResponse {
	resp := SimplifyResponse{Result: make([]SimplifyReplacement, 0)}
	for _, response := range responses {
		resp.Result = append(resp.Result, response.Result...)
	}

	return resp
}

type AnalyticsRequest struct {
	Blocks []string `json:"blocks"`
	Url    string   `json:"url"`
}

type AnalyzeChunckAI struct {
	Total           int `json:"total_words"`
	AggressiveCount int `json:"aggressive_words"`
	AggitationCount int `json:"aggitation_words"`
	MatCount        int `json:"mat_words"`
	BiasCount       int `json:"bias_words"`
}

type AnalyzeResponse struct {
	AggressiveWordsPercent int `json:"aggressive_percent"`
	AggitationWordsPercent int `json:"aggitation_percent"`
	MatWordsPercent        int `json:"mat_percent"`
	BiasWordsPercent       int `json:"bias_percent"`
}

type AnalyzeResponseAPI struct {
	AggressiveWordsPercent int    `json:"aggressive_percent"`
	AggitationWordsPercent int    `json:"aggitation_percent"`
	MatWordsPercent        int    `json:"mat_percent"`
	BiasWordsPercent       int    `json:"bias_percent"`
	Resume                 string `json:"resume"`
}

func (a *AI) SaveAnalytics(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()

	var req AnalyticsRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		utils.LogError(ctx, err, utils.MsgErrUnmarshalRequest)
		http.Error(w, utils.Invalid, http.StatusBadRequest)
		return
	}

	err := a.hasAnalyticsForURL(ctx, req.Url)
	if err != nil {
		fmt.Printf("[DEBUG] answer from postgres", err)
		if err := json.NewEncoder(w).Encode("ok"); err != nil {
			utils.LogError(ctx, err, utils.MsgErrMarshalResponse)
			http.Error(w, utils.Internal, http.StatusInternalServerError)
			return
		}
		return
	} else {
		fmt.Printf("[DEBUG] no analytics for url in postgres")
	}

	chunks := utils.SplitBlocksIntoChunks(req.Blocks, a.cfg.SimplifyMaxTokensInChunk)
	wg := &sync.WaitGroup{}
	responsesCh := make(chan AnalyzeChunckAI)
	errorsCh := make(chan error)

	for i, chunk := range chunks {
		wg.Add(1)
		go a.sendChunkRequestAnalytics(ctx, analyzeTextPrompt, chunk, i, wg, responsesCh, errorsCh)
	}

	go func() {
		wg.Wait()
		close(responsesCh)
		close(errorsCh)
	}()

	chunkResponses := make([]AnalyzeChunckAI, 0)
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

	resp := calculateResponsesAnalyze(chunkResponses)
	if err := a.insertAnalytics(ctx, resp, req.Url); err != nil {
		utils.LogError(ctx, err, "fail to insert analytics data")
		http.Error(w, utils.Internal, http.StatusInternalServerError)
		return
	}
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		utils.LogError(ctx, err, utils.MsgErrMarshalResponse)
		http.Error(w, utils.Internal, http.StatusInternalServerError)
		return
	}
}

func calculateResponsesAnalyze(responses []AnalyzeChunckAI) AnalyzeResponse {
	common := AnalyzeChunckAI{}
	for _, response := range responses {
		common.Total += response.Total
		common.AggressiveCount += response.AggressiveCount
		common.AggitationCount += response.AggitationCount
		common.MatCount += response.MatCount
		common.BiasCount += response.BiasCount
	}

	resp := AnalyzeResponse{
		AggressiveWordsPercent: 100 * common.AggressiveCount / common.Total,
		AggitationWordsPercent: 100 * common.AggitationCount / common.Total,
		MatWordsPercent:        100 * common.MatCount / common.Total,
		BiasWordsPercent:       100 * common.BiasCount / common.Total,
	}

	return resp
}

func (a *AI) sendChunkRequestAnalytics(ctx context.Context, promptFormat string, chunkData string, index int, wg *sync.WaitGroup, responses chan<- AnalyzeChunckAI, errorsCh chan<- error) {
	defer wg.Done()

	defaultResp := AnalyzeChunckAI{}

	chunkBytes, err := json.Marshal(chunkData)
	if err != nil {
		errorsCh <- errors.Wrapf(err, "failed to marshal chunk slice")
		responses <- defaultResp
	}
	chunk := string(chunkBytes)

	chunkToAI := replacerReq.Replace(chunk)
	prompt := fmt.Sprintf(promptFormat, chunkToAI)
	answer, err := a.SendRequest(prompt)
	if err != nil {
		errorsCh <- errors.Wrapf(err, "failed to send chunk request, index=%d", index)
		return
	}

	answer = strings.TrimPrefix(answer, "```")
	answer = strings.TrimPrefix(answer, "json")
	answer = strings.TrimSuffix(answer, "```")
	answer = replacerResp.Replace(answer)

	fmt.Printf("[DEBUG] answer=%s\n", answer)

	var resp AnalyzeChunckAI
	if err = json.Unmarshal([]byte(answer), &resp); err != nil {
		errorsCh <- errors.Wrapf(err, "failed to unmarshal chunk response, index=%d", index)
		return
	}

	responses <- resp
}

func (a *AI) GetAnalytics(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()

	url := r.URL.Query().Get("url")
	if url == "" {
		utils.LogError(ctx, errors.New("empty query param"), `query param "url" is empty`)
		http.Error(w, utils.Invalid, http.StatusBadRequest)
		return
	}

	if index := strings.Index(url, "/"); index != -1 {
		url = url[:index]
	}
	stat, err := a.getAnalyticsForURL(ctx, url)
	if err != nil {
		utils.LogError(ctx, err, "fail to get analytics for url")
		http.Error(w, utils.Internal, http.StatusInternalServerError)
		return
	}
	res := calculateAnalyzeStat(stat)
	if err := json.NewEncoder(w).Encode(res); err != nil {
		utils.LogError(ctx, err, utils.MsgErrMarshalResponse)
		http.Error(w, utils.Internal, http.StatusInternalServerError)
		return
	}
}

func calculateAnalyzeStat(responses []AnalyzeResponse) AnalyzeResponseAPI {
	common := AnalyzeResponse{}
	length := len(responses)
	for _, response := range responses {
		common.AggressiveWordsPercent += response.AggressiveWordsPercent
		common.AggitationWordsPercent += response.AggitationWordsPercent
		common.MatWordsPercent += response.MatWordsPercent
		common.BiasWordsPercent += response.BiasWordsPercent
	}
	resume := "рекомендован"
	if common.AggressiveWordsPercent > 15 || common.AggitationWordsPercent > 15 || common.MatWordsPercent > 10 || common.BiasWordsPercent > 20 {
		resume = "не " + resume
	}
	return AnalyzeResponseAPI{
		AggressiveWordsPercent: common.AggressiveWordsPercent / length,
		AggitationWordsPercent: common.AggitationWordsPercent / length,
		MatWordsPercent:        common.MatWordsPercent / length,
		BiasWordsPercent:       common.BiasWordsPercent / length,
		Resume:                 resume,
	}
}

func (a *AI) insertAnalytics(ctx context.Context, data AnalyzeResponse, url string) error {
	query := `INSERT INTO url_analytics(url, aggressive_percent, aggitation_percent, mat_percent, bias_percent)
				VALUES ($1, $2, $3, $4, $5);`
	_, err := a.db.Exec(ctx, query, url, data.AggressiveWordsPercent,
		data.AggitationWordsPercent, data.MatWordsPercent, data.BiasWordsPercent)
	if err != nil {
		return err
	}
	return nil
}

func (a *AI) getAnalyticsForURL(ctx context.Context, url string) ([]AnalyzeResponse, error) {
	res := make([]AnalyzeResponse, 0)
	query := `SELECT aggressive_percent, aggitation_percent, mat_percent, bias_percent 
				FROM url_analytics WHERE url LIKE $1`
	rows, err := a.db.Query(ctx, query, url+"%")
	if err != nil {
		return res, err
	}
	defer rows.Close()
	for rows.Next() {
		var r AnalyzeResponse
		err := rows.Scan(&r.AggressiveWordsPercent, &r.AggitationWordsPercent, &r.MatWordsPercent, &r.BiasWordsPercent)
		if err != nil {
			return []AnalyzeResponse{}, errors.Wrap(err, "failed to parse animal")
		}
		res = append(res, r)
	}
	return res, nil
}

func (a *AI) hasAnalyticsForURL(ctx context.Context, url string) error {
	query := `SELECT id FROM url_analytics WHERE url=$1`
	var id int
	err := a.db.QueryRow(ctx, query, url).Scan(&id)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil
		}
		return err
	}
	return nil
}
