package chatgpt

import (
	"encoding/json"
	"net/http"

	"purify/src/config"
	"purify/src/utils"
)

type ChatGPT struct {
	cfg config.ChatGPTConfig
}

func NewChatGPT(cfg config.ChatGPTConfig) *ChatGPT {
	return &ChatGPT{cfg: cfg}
}

type ReplaceTextRequest struct {
	Prompt string `json:"prompt"`
}

type ReplaceTextResponse struct {
	Answer string `json:"answer"`
}

func (c *ChatGPT) ReplaceText(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()

	var req ReplaceTextRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		utils.LogError(ctx, err, utils.MsgErrUnmarshalRequest)
		http.Error(w, utils.Invalid, http.StatusBadRequest)
		return
	}

	answer, err := sendRequest(req.Prompt, c.cfg)
	if err != nil {
		utils.LogError(ctx, err, "sendRequest error")
		http.Error(w, utils.Internal, http.StatusInternalServerError)
		return
	}

	resp := ReplaceTextResponse{Answer: answer}
	if err = json.NewEncoder(w).Encode(resp); err != nil {
		utils.LogError(ctx, err, utils.MsgErrMarshalResponse)
		http.Error(w, utils.Internal, http.StatusInternalServerError)
		return
	}
}
