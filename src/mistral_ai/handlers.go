package mistral_ai

import (
	"encoding/json"
	"net/http"

	"purify/src/config"
	"purify/src/utils"
)

type MistralAI struct {
	cfg config.MistralAIConfig
}

func NewMistralAI(cfg config.MistralAIConfig) *MistralAI {
	return &MistralAI{cfg: cfg}
}

type AskRequest struct {
	Text string `json:"text"`
}

type AskResponse struct {
	Response string `json:"response"`
}

func (m *MistralAI) Ask(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()

	var req AskRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		utils.LogError(ctx, err, utils.MsgErrUnmarshalRequest)
		http.Error(w, utils.Invalid, http.StatusBadRequest)
		return
	}

	result, err := sendMistralAIRequest(req.Text, m.cfg)
	if err != nil {
		utils.LogError(ctx, err, "sendMistralAIRequest error")
		http.Error(w, utils.Internal, http.StatusInternalServerError)
		return
	}

	resp := AskResponse{Response: result}
	if err = json.NewEncoder(w).Encode(resp); err != nil {
		utils.LogError(r.Context(), err, utils.MsgErrMarshalResponse)
		http.Error(w, utils.Internal, http.StatusInternalServerError)
		return
	}
}
