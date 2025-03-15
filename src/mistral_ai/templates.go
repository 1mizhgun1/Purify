package mistral_ai

var requestTemplate = `{"model": "mistral-large-latest", "messages": [{"role": "user", "content": "%s"}]}`

type MistralAIResponse struct {
	Choices []Choice `json:"choices"`
}
type Choice struct {
	Index   int     `json:"index"`
	Message Message `json:"message"`
}

type Message struct {
	Content string `json:"content"`
}
