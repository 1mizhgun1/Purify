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

var analyzePromptTemplate = "Проанализируй следующий текст и определи части текста, которые содержат агрессию (обозначай числом 1) и предвзятость (числом 2). Возвращай результат в формате JSON, где каждая часть текста сопровождается соответствующим значением state. Пример Входной текст: \\\"Привет, вы все говно. Пушкин плохо писал стихи.\\\" Выходной формат: [{\\\"text\\\":\\\"Привет, \\\",\\\"state\\\":0},{\\\"text\\\":\\\"вы все говно. \\\",\\\"state\\\":1},{\\\"text\\\":\\\"Пушкин плохо писал стихи.\\\",\\\"state\\\":2}] Текст для анализа: \\\"%s\\\" Напиши только сам ответ и ничего лишнего"
