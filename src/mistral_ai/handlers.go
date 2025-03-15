package mistral_ai

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"sync"

	"github.com/pkg/errors"
	"purify/src/config"
	"purify/src/utils"
)

type MistralAI struct {
	cfg config.MistralAIConfig
}

func NewMistralAI(cfg config.MistralAIConfig) *MistralAI {
	return &MistralAI{cfg: cfg}
}

type AnalyzeTextRequest struct {
	Text string `json:"text"`
}

type AnalyzeTextResponse struct {
	Response []TextPart `json:"response"`
}

type TextPart struct {
	Text  string `json:"text"`
	State int    `json:"state"`
}

func (m *MistralAI) AnalyzeText(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()

	var req AnalyzeTextRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		utils.LogError(ctx, err, utils.MsgErrUnmarshalRequest)
		http.Error(w, utils.Invalid, http.StatusBadRequest)
		return
	}

	chunks := utils.SplitTextIntoChunks(req.Text, m.cfg.WordsInChunk)

	wg := &sync.WaitGroup{}
	responses := make(chan ChunkInChannel)
	errorsCh := make(chan error)

	for i, chunk := range chunks {
		wg.Add(1)
		go m.sendChunkRequest(chunk, i, wg, responses, errorsCh)
	}

	go func() {
		wg.Wait()
		close(responses)
		close(errorsCh)
	}()

	chunkResponses := make([]ChunkInChannel, 0)
	respErrors := make([]error, 0)

	for {
		select {
		case response, ok := <-responses:
			if !ok {
				goto end
			}
			chunkResponses = append(chunkResponses, response)
		case err, ok := <-errorsCh:
			if !ok {
				continue
			}
			respErrors = append(respErrors, err)
		}
	}

end:
	if len(respErrors) > 0 {
		utils.LogErrorMessage(ctx, fmt.Sprintf("got errors: %v", respErrors))
	}

	resp := AnalyzeTextResponse{Response: joinResponses(chunkResponses)}
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		utils.LogError(r.Context(), err, utils.MsgErrMarshalResponse)
		http.Error(w, utils.Internal, http.StatusInternalServerError)
		return
	}
}

type ChunkInChannel struct {
	Index int
	Parts []TextPart
}

func (m *MistralAI) sendChunkRequest(chunk string, index int, wg *sync.WaitGroup, responses chan<- ChunkInChannel, errorsCh chan<- error) {
	defer wg.Done()

	prompt := fmt.Sprintf(analyzePromptTemplate, chunk)
	resultMessage, err := sendMistralAIRequest(prompt, m.cfg)
	if err != nil {
		responses <- ChunkInChannel{Index: index, Parts: []TextPart{{Text: chunk, State: 0}}}
		errorsCh <- errors.Wrapf(err, "failed to send chunk request, index=%d", index)
		return
	}

	fmt.Printf("[DEBUG] i=%d resultMessage=%s\n", index, resultMessage)

	resultMessage = strings.TrimPrefix(resultMessage, "```json")
	resultMessage = strings.TrimSuffix(resultMessage, "```")

	var textParts []TextPart
	if err = json.Unmarshal([]byte(resultMessage), &textParts); err != nil {
		responses <- ChunkInChannel{Index: index, Parts: []TextPart{{Text: chunk, State: 0}}}
		errorsCh <- errors.Wrapf(err, "failed to unmarshal chunk response, index=%d", index)
		return
	}

	responses <- ChunkInChannel{Index: index, Parts: textParts}
}

func joinResponses(responses []ChunkInChannel) []TextPart {
	sort.SliceStable(responses, func(i, j int) bool {
		return responses[i].Index < responses[j].Index
	})

	result := make([]TextPart, 0)
	for _, chunkResponse := range responses {
		result = append(result, chunkResponse.Parts...)
	}

	return result
}
