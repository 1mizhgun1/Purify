package mistral_ai

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

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

	prompt := fmt.Sprintf(analyzePromptTemplate, req.Text)
	resultMessage, err := sendMistralAIRequest(prompt, m.cfg)
	if err != nil {
		utils.LogError(ctx, err, "sendMistralAIRequest error")
		http.Error(w, utils.Internal, http.StatusInternalServerError)
		return
	}

	fmt.Println("[DEBUG] resultMessage = " + resultMessage)

	resultMessage = strings.TrimPrefix(resultMessage, "```json")
	resultMessage = strings.TrimSuffix(resultMessage, "```")

	var textParts []TextPart
	if err = json.Unmarshal([]byte(resultMessage), &textParts); err != nil {
		utils.LogError(ctx, err, "incorrect response format from MistralAI")
		http.Error(w, utils.Internal, http.StatusInternalServerError)
		return
	}

	resp := AnalyzeTextResponse{Response: textParts}
	if err = json.NewEncoder(w).Encode(resp); err != nil {
		utils.LogError(r.Context(), err, utils.MsgErrMarshalResponse)
		http.Error(w, utils.Internal, http.StatusInternalServerError)
		return
	}
}
